// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OpenKind",
    platforms: [.macOS(.v12), .iOS(.v15)],
    products: [.library(name: "OpenKind", targets: ["OpenKind"])],
    targets: [
        .target(name: "OpenKind"),
        .testTarget(name: "OpenKindTests", dependencies: ["OpenKind"]),
    ]
)
