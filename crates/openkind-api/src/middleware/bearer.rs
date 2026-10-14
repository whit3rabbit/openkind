//! One `Authorization: Bearer` parser shared by the HTTP gate, the gRPC
//! check, and the proxy's credential forwarding, so the three cannot accept
//! different spellings of the same header.

/// Extract the token from an `Authorization` header value.
///
/// Policy (RFC 9110 for the scheme, strict for the rest):
/// - the scheme `Bearer` is matched case-insensitively;
/// - exactly one ASCII space separates scheme and token;
/// - the token is returned exactly as sent, never trimmed, so trailing
///   whitespace makes the credential fail comparison instead of being
///   silently repaired;
/// - an empty token, or one that starts with more whitespace, is rejected.
pub(crate) fn parse_bearer(value: &str) -> Option<&str> {
    let (scheme, token) = value.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("Bearer") {
        return None;
    }
    if token.is_empty() || token.starts_with(|c: char| c.is_ascii_whitespace()) {
        return None;
    }
    Some(token)
}

#[cfg(test)]
mod tests {
    use super::parse_bearer;

    #[test]
    fn scheme_is_case_insensitive() {
        for scheme in ["Bearer", "bearer", "BEARER", "bEaReR"] {
            assert_eq!(parse_bearer(&format!("{scheme} tok")), Some("tok"));
        }
    }

    #[test]
    fn token_is_taken_verbatim() {
        assert_eq!(parse_bearer("Bearer a.b-c_d=="), Some("a.b-c_d=="));
        assert_eq!(parse_bearer("Bearer tok "), Some("tok "), "not trimmed");
    }

    #[test]
    fn malformed_values_are_rejected() {
        for value in [
            "",
            "Bearer",
            "Bearer ",
            "Bearer  tok",
            "Bearer\ttok",
            "Bearer\ttok ",
            "Bearer \ttok",
            "Basic dXNlcjpwYXNz",
            "Bearerx tok",
            " Bearer tok",
            "tok",
        ] {
            assert_eq!(parse_bearer(value), None, "{value:?}");
        }
    }
}
