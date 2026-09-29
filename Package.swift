// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OpenKind",
    platforms: [.macOS(.v12), .iOS(.v15)],
    products: [.library(name: "OpenKind", targets: ["OpenKind"])],
    targets: [
        .target(
            name: "OpenKind",
            path: "bindings/swift/Sources/OpenKind"
        ),
        .testTarget(
            name: "OpenKindTests",
            dependencies: ["OpenKind"],
            path: "bindings/swift/Tests/OpenKindTests"
        ),
    ]
)
