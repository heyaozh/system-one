//! Derive macros for the `jev` crate.
//!
//! * `#[derive(JevChoice)]` on a unit-variant enum → the options of a `choice`.
//! * `#[derive(JevQuestions)]` on a struct of `Noul` / `Choice<E>` / `Score`
//!   fields → a question schema and a typed decoder.
//!
//! See the `jev` crate documentation for usage; this crate is not meant to
//! be used directly.

use heck::ToSnakeCase;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    parse_macro_input, Attribute, Data, DeriveInput, Expr, ExprArray, ExprLit, Fields, GenericArgument, Ident, Lit,
    LitStr, Meta, PathArguments, Token, Type,
};

// ---------------------------------------------------------------------------
// #[derive(JevChoice)]
// ---------------------------------------------------------------------------

/// Derive the options of a `choice` question from a unit-variant enum.
///
/// The enum must also derive (or implement) `Clone, Copy, PartialEq, Debug`.
///
/// Attributes on variants:
/// * `#[jev(key = "wire_key")]` — override the option key (default: snake_case variant name).
/// * `#[jev(desc = "…")]` or a `///` doc comment — description sent to the backend.
#[proc_macro_derive(JevChoice, attributes(jev))]
pub fn derive_jev_choice(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_choice(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_choice(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(name, "JevChoice can only be derived for enums"));
    };
    if data.variants.len() > 255 {
        return Err(syn::Error::new_spanned(name, "a choice may have at most 255 options"));
    }
    if data.variants.is_empty() {
        return Err(syn::Error::new_spanned(name, "a choice needs at least one option"));
    }

    let mut idents = Vec::new();
    let mut keys = Vec::new();
    let mut descs = Vec::new();
    for v in &data.variants {
        if !matches!(v.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(&v.ident, "JevChoice variants must be unit variants"));
        }
        let args = JevArgs::from_attrs(&v.attrs)?;
        let key = args.key.unwrap_or_else(|| v.ident.to_string().to_snake_case());
        let desc = args.desc.or_else(|| doc_comment(&v.attrs));
        idents.push(&v.ident);
        keys.push(key);
        descs.push(match desc {
            Some(d) => quote! { ::core::option::Option::Some(#d) },
            None => quote! { ::core::option::Option::None },
        });
    }

    Ok(quote! {
        impl ::jev::__private::JevChoice for #name {
            fn all() -> &'static [Self] {
                const ALL: &[#name] = &[ #( #name::#idents ),* ];
                ALL
            }
            fn key(&self) -> &'static str {
                match self { #( #name::#idents => #keys ),* }
            }
            fn description(&self) -> ::core::option::Option<&'static str> {
                match self { #( #name::#idents => #descs ),* }
            }
            fn from_key(key: &str) -> ::core::option::Option<Self> {
                match key { #( #keys => ::core::option::Option::Some(#name::#idents), )* _ => ::core::option::Option::None }
            }
        }
    })
}

// ---------------------------------------------------------------------------
// #[derive(JevQuestions)]
// ---------------------------------------------------------------------------

/// Derive a question schema and decoder from a struct.
///
/// Every field must be `Noul`, `Choice<E>` (with `E: JevChoice`) or `Score`.
///
/// Field attributes (`#[jev(...)]`):
/// * `"instructions"` (positional) or `instructions = "…"` — the question text (required).
/// * `levels = ["Low", "Mid", "High"]` — for `Score`: 2–10 ordered level descriptions (required).
/// * `yes = "…"`, `no = "…"` — for `Noul`: optional explicit criteria.
/// * `name = "wire_name"` — override the question name (default: field name).
#[proc_macro_derive(JevQuestions, attributes(jev))]
pub fn derive_jev_questions(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_questions(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

#[allow(clippy::large_enum_variant)]
enum FieldKind {
    Noul,
    Choice(Type),
    Score,
}

fn expand_questions(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(name, "JevQuestions can only be derived for structs"));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(name, "JevQuestions needs named fields"));
    };

    let mut schema_inserts = Vec::new();
    let mut field_decoders = Vec::new();
    let mut helpers: Vec<TokenStream2> = Vec::new();

    for f in &fields.named {
        let ident = f.ident.as_ref().unwrap();
        let args = JevArgs::from_attrs(&f.attrs)?;
        let wire = args.name.clone().unwrap_or_else(|| ident.to_string());
        let instructions = args
            .instructions
            .clone()
            .or_else(|| doc_comment(&f.attrs))
            .ok_or_else(|| syn::Error::new_spanned(ident, "missing #[jev(\"instructions\")] (or a doc comment)"))?;

        match field_kind(&f.ty)? {
            FieldKind::Noul => {
                let criteria = match (&args.yes, &args.no) {
                    (Some(y), Some(n)) => quote! {
                        ::core::option::Option::Some(::jev::__private::NoulCriteria { yes: #y.to_string(), no: #n.to_string() })
                    },
                    (None, None) => quote! { ::core::option::Option::None },
                    _ => return Err(syn::Error::new_spanned(ident, "`yes` and `no` must be given together")),
                };
                schema_inserts.push(quote! {
                    schema.insert(#wire, ::jev::__private::QuestionSpec::Noul {
                        instructions: #instructions.to_string(),
                        criteria: #criteria,
                    });
                });
                field_decoders.push(quote! {
                    #ident: ::jev::__private::Noul::from_raw(#wire, raw.get(#wire)?)?,
                });
            }
            FieldKind::Choice(enum_ty) => {
                schema_inserts.push(quote! {
                    schema.insert(#wire, ::jev::__private::QuestionSpec::Choice {
                        instructions: #instructions.to_string(),
                        criteria: <#enum_ty as ::jev::__private::JevChoice>::criteria(),
                    });
                });
                field_decoders.push(quote! {
                    #ident: ::jev::__private::Choice::<#enum_ty>::from_raw(#wire, raw.get(#wire)?)?,
                });
            }
            FieldKind::Score => {
                let levels = args
                    .levels
                    .clone()
                    .ok_or_else(|| syn::Error::new_spanned(ident, "Score fields need #[jev(levels = [\"…\", …])]"))?;
                if levels.len() < 2 || levels.len() > 10 {
                    return Err(syn::Error::new_spanned(ident, "a score needs 2 to 10 levels"));
                }
                let levels_fn = format_ident!("__jev_levels_{}", ident);
                schema_inserts.push(quote! {
                    schema.insert(#wire, ::jev::__private::QuestionSpec::Score {
                        instructions: #instructions.to_string(),
                        criteria: Self::#levels_fn(),
                    });
                });
                field_decoders.push(quote! {
                    #ident: ::jev::__private::Score::from_raw(#wire, raw.get(#wire)?, &Self::#levels_fn())?,
                });
                // A tiny helper so schema() and from_raw() share one source of truth.
                helpers.push(quote! {
                    #[doc(hidden)]
                    fn #levels_fn() -> ::std::vec::Vec<::std::string::String> {
                        vec![ #( #levels.to_string() ),* ]
                    }
                });
            }
        }
    }

    Ok(quote! {
        impl #name {
            #( #helpers )*
        }
        impl ::jev::__private::JevQuestions for #name {
            fn schema() -> ::jev::__private::QuestionSchema {
                let mut schema = ::jev::__private::QuestionSchema::new();
                #( #schema_inserts )*
                schema
            }
            fn from_raw(raw: &::jev::__private::RawAnswers) -> ::jev::__private::Result<Self> {
                ::core::result::Result::Ok(Self {
                    #( #field_decoders )*
                })
            }
        }
    })
}

fn field_kind(ty: &Type) -> syn::Result<FieldKind> {
    let Type::Path(tp) = ty else {
        return Err(syn::Error::new_spanned(ty, "field type must be Noul, Choice<E> or Score"));
    };
    let seg = tp.path.segments.last().unwrap();
    match seg.ident.to_string().as_str() {
        "Noul" => Ok(FieldKind::Noul),
        "Score" => Ok(FieldKind::Score),
        "Choice" => {
            let PathArguments::AngleBracketed(ab) = &seg.arguments else {
                return Err(syn::Error::new_spanned(ty, "Choice needs a type parameter: Choice<MyEnum>"));
            };
            match ab.args.first() {
                Some(GenericArgument::Type(t)) => Ok(FieldKind::Choice(t.clone())),
                _ => Err(syn::Error::new_spanned(ty, "Choice needs a type parameter: Choice<MyEnum>")),
            }
        }
        other => Err(syn::Error::new_spanned(ty, format!("unsupported field type `{other}`; use Noul, Choice<E> or Score"))),
    }
}

// ---------------------------------------------------------------------------
// #[jev(...)] attribute parsing
// ---------------------------------------------------------------------------

#[derive(Default)]
struct JevArgs {
    instructions: Option<String>,
    levels: Option<Vec<String>>,
    yes: Option<String>,
    no: Option<String>,
    name: Option<String>,
    key: Option<String>,
    desc: Option<String>,
}

enum Arg {
    Positional(LitStr),
    Named(Ident, Expr),
}

impl Parse for Arg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(LitStr) {
            return Ok(Arg::Positional(input.parse()?));
        }
        let ident: Ident = input.parse()?;
        let _eq: Token![=] = input.parse()?;
        let expr: Expr = input.parse()?;
        Ok(Arg::Named(ident, expr))
    }
}

impl JevArgs {
    fn from_attrs(attrs: &[Attribute]) -> syn::Result<Self> {
        let mut out = JevArgs::default();
        for attr in attrs.iter().filter(|a| a.path().is_ident("jev")) {
            let Meta::List(list) = &attr.meta else {
                return Err(syn::Error::new_spanned(attr, "expected #[jev(...)]"));
            };
            let args = list.parse_args_with(Punctuated::<Arg, Token![,]>::parse_terminated)?;
            for a in args {
                match a {
                    Arg::Positional(s) => out.instructions = Some(s.value()),
                    Arg::Named(id, expr) => match id.to_string().as_str() {
                        "instructions" => out.instructions = Some(expr_str(&expr)?),
                        "levels" => out.levels = Some(expr_str_array(&expr)?),
                        "yes" => out.yes = Some(expr_str(&expr)?),
                        "no" => out.no = Some(expr_str(&expr)?),
                        "name" => out.name = Some(expr_str(&expr)?),
                        "key" => out.key = Some(expr_str(&expr)?),
                        "desc" => out.desc = Some(expr_str(&expr)?),
                        other => {
                            return Err(syn::Error::new_spanned(
                                id,
                                format!("unknown jev attribute `{other}` (expected instructions, levels, yes, no, name, key, desc)"),
                            ))
                        }
                    },
                }
            }
        }
        Ok(out)
    }
}

fn expr_str(e: &Expr) -> syn::Result<String> {
    match e {
        Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) => Ok(s.value()),
        _ => Err(syn::Error::new_spanned(e, "expected a string literal")),
    }
}

fn expr_str_array(e: &Expr) -> syn::Result<Vec<String>> {
    match e {
        Expr::Array(ExprArray { elems, .. }) => elems.iter().map(expr_str).collect(),
        _ => Err(syn::Error::new_spanned(e, "expected an array of string literals: [\"a\", \"b\"]")),
    }
}

/// Join `///` doc comment lines into one description.
fn doc_comment(attrs: &[Attribute]) -> Option<String> {
    let mut lines = Vec::new();
    for a in attrs.iter().filter(|a| a.path().is_ident("doc")) {
        if let Meta::NameValue(nv) = &a.meta {
            if let Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) = &nv.value {
                lines.push(s.value().trim().to_string());
            }
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join(" "))
    }
}
