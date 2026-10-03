import Foundation
import XCTest
import OpenKind

final class OpenKindAPIKeyTests: XCTestCase {
    func testGeneratedKeysHaveTheSharedFormatAndAreDistinct() throws {
        let keys = try (0..<8).map { _ in try OpenKindAPIKey.generate() }
        for key in keys {
            XCTAssertEqual(key.utf8.count, 67)
            XCTAssertNotNil(key.range(of: "^ok_[0-9a-f]{64}$", options: .regularExpression))
        }
        XCTAssertEqual(Set(keys).count, keys.count)
    }

#if os(macOS)
    func testServerRejectsInvalidKeysWithoutEchoingThem() {
        let invalid = ["", " ", " secret", "secret ", "secret key", "secret\tkey", "secret\nkey",
                       "secret\rkey", "secret\u{0}", "secret\u{7f}", "key\u{e9}", "key\u{1f600}"]
        for key in invalid {
            XCTAssertThrowsError(try OpenKindServer(apiKey: key)) { error in
                guard case ServerError.invalidConfiguration(let message) = error else {
                    return XCTFail("expected an invalid configuration error")
                }
                XCTAssertEqual(message, "apiKey must be nonempty visible ASCII without whitespace")
            }
        }
    }

    func testServerAllowsNoKeyAndVisibleASCIIKeys() throws {
        XCTAssertNil(try OpenKindServer().apiKey)
        let key = String((0x21...0x7e).map { Character(UnicodeScalar($0)!) })
        XCTAssertEqual(try OpenKindServer(apiKey: key).apiKey, key)
    }
#endif
}
