# Swift library

Swift Package Manager library for macOS 12+ and iOS 15+. The repository root and `bindings/swift` both expose the `OpenKind` product and `OpenKind` module. Use the repository root as a package dependency from another checkout. Run `swift test` from this directory for the local contract tests. The package is not published to a Swift package registry.

For the sibling TurboSpark checkout, add the OpenKind package and product to `swift/TurboSparkApp/Package.swift`:

```swift
.package(path: "../../../openkind")
```

```swift
.product(name: "OpenKind", package: "openkind")
```

The relative path assumes `openkind` and `turbospark` are sibling directories. For a remote dependency, use the public Git repository URL and a semantic version tag. The repository URL and first release tag are not set in this checkout yet.

```swift
import Foundation
import OpenKind

let client = OpenKindClient(
    baseURL: URL(string: "http://127.0.0.1:18080")!
)
let result = try await client.systemOne(
    state: .string("I was charged twice."),
    questions: ["billing": .noul(instructions: .string("Is this a billing issue?"))],
    model: "mock"
)
print(result.data.answers, result.requestID as Any)
```

`evaluate(_:)` accepts an explicit `SystemRequest`, `listModels()` lists registered aliases, and `health()` probes the daemon without auth. `ApiError` carries the HTTP status, error code, and request ID. `ClientError.invalidResponse` marks request-inconsistent success responses. Set `OPENKIND_TEST_URL` and, if needed, `OPENKIND_TEST_API_KEY` before `swift test` to include the live daemon test. See the [shared scope](../README.md).

## API keys

Generate a key explicitly on macOS or iOS:

```swift
let apiKey = try OpenKindAPIKey.generate()
```

The helper uses operating-system randomness to produce an `ok_` prefix and 64 lowercase hexadecimal characters (256 bits). It throws if randomness is unavailable and does not save the key or start a daemon.

Authentication is optional: `OpenKindServer` defaults to `apiKey: nil`. Supply a generated or existing key to enable it. Server keys must be nonempty visible ASCII with no whitespace; invalid values throw `ServerError.invalidConfiguration`. Keys are not persisted to `~/.openkind` or another configuration file, so supply the key again when recreating the server.

When connecting to an existing service, the client's `apiKey` must match that service's configured key, for example `OpenKindClient(baseURL: url, apiKey: "server-configured-key")`. Generating a new key does not configure the remote service. The client reads no environment variables.

## Start a local server on macOS

Build `openkindd` first. `OpenKindServer` is available on macOS; iOS applications use `OpenKindClient` with an existing service.

```swift
let apiKey = try OpenKindAPIKey.generate()
let server = try OpenKindServer(
    binary: "/path/to/openkindd",
    httpAddress: "127.0.0.1:18080",
    models: ["mock"],
    apiKey: apiKey
)
try await server.start()
do {
    let models = try await server.client.listModels()
    print(models.data.models)
} catch {
    await server.stop()
    throw error
}
await server.stop()
```

The wrapper binds HTTP to loopback, disables gRPC, and terminates only its child process.
