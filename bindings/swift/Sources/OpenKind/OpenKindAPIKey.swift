import Foundation
import Security

/// Creates API keys without starting a daemon or saving configuration.
public enum OpenKindAPIKey {
    /// Generate an `ok_` key containing 256 bits of operating-system randomness.
    public static func generate() throws -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = bytes.withUnsafeMutableBytes {
            SecRandomCopyBytes(kSecRandomDefault, $0.count, $0.baseAddress!)
        }
        guard status == errSecSuccess else {
            throw NSError(
                domain: "OpenKindAPIKey",
                code: Int(status),
                userInfo: [NSLocalizedDescriptionKey: "Could not generate an API key from operating-system randomness"]
            )
        }
        return "ok_" + bytes.map { String(format: "%02x", $0) }.joined()
    }
}
