# Puppy 家长端

iPhone / iPad SwiftUI 客户端，部署目标 iOS 17.0。使用 Xcode 打开 `PuppyParent.xcodeproj`，选择 `PuppyParent` scheme 和一台可用的 iOS 模拟器，执行 Product → Test。测试夹具包含合成协议回执，离线演示可以从连接页打开。

```sh
xcodebuild -project parent-ios/PuppyParent.xcodeproj -scheme PuppyParent \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -parallel-testing-enabled NO CODE_SIGNING_ALLOWED=NO test
```

命令从仓库根目录执行，模拟器名称按本机已安装的运行时调整。实体机安装需要使用者自己的签名和配对设置。候选版使用系统默认应用图标。
