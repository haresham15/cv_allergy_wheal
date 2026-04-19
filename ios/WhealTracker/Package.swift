// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "WhealTracker",
    platforms: [
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "WhealTracker",
            targets: ["WhealTracker"]
        ),
    ],
    dependencies: [
        // No external dependencies. Pure Apple native pipeline.
    ],
    targets: [
        .target(
            name: "WhealTracker",
            dependencies: [],
            path: "WhealTracker"
        ),
    ]
)
