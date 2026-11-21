# RUG Puppy

[项目宣传页](https://scott987-cmd.github.io/RUG-Puppy/) · [开发者预览版](https://github.com/scott987-cmd/RUG-Puppy/releases/tag/v0.1.0-preview.1)

Developer preview of an ESP32-S3 robot companion with a local macOS gateway and a SwiftUI parent client. The gateway handles conversation, stories, English practice, and on-demand camera observations. This preview enables stationary play in the supplied Puppy build defaults and gateway example.

| Directory | Contents |
| --- | --- |
| `main/`, `partitions/`, `tools/` | ESP-IDF firmware, flash layout, and build tools |
| `companion/` | Local gateway, browser parent panel, and Python regression suite |
| `parent-ios/` | Xcode project and iOS unit/UI tests |
| `components/78__esp-ml307/` | Local networking component override |
| `scripts/` | Host regression, optional speech probe, and isolated development worker |

- [构建、配置与测试](docs/BUILD.md)
- [iOS 家长端](parent-ios/README.md)
- [构建与测试结果](docs/VALIDATION.md)
- [许可证](LICENSE) · [上游与第三方署名](NOTICE)

This is a source preview. Running the host tests does not qualify unsupervised child use. Moving while listening for quiet stop commands, low-light recognition, and directional sound search remain experimental or pending hardware acceptance.
