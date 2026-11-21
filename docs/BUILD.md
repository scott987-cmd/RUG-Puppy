# 构建、配置与测试

## 获取源码

需要 Git。获取包含最新文档和宣传页修复的源码：

```sh
git clone https://github.com/scott987-cmd/RUG-Puppy.git
cd RUG-Puppy
```

[开发者预览版下载](https://github.com/scott987-cmd/RUG-Puppy/releases/tag/v0.1.0-preview.1)中的 `rug-puppy-open-source-candidate.tar.gz` 是首发已验证快照。把它、`FILES.tsv` 和 `SHA256SUMS` 放在同一个目录，先校验再解压。GitHub 自动生成的 Source code 归档内容包含额外的宣传页和发布文档，适合查看整个发布提交。

```sh
shasum -a 256 -c SHA256SUMS
tar -xzf rug-puppy-open-source-candidate.tar.gz
cd rug-puppy
```

首发快照适用本文固件与网关步骤。最新文档在 `main` 分支。

## ESP32-S3 固件

使用 ESP-IDF **5.5.5**。安装目标芯片工具链，构建时让 ESP-IDF Component Manager 解析 `main/idf_component.yml`。项目自带的 `components/78__esp-ml307` 会作为本地覆盖。建议用不含空格的构建目录。此版构建实际解析到的组件版本与校验值见 [ESP-IDF 依赖清单](ESP-IDF-components.tsv)。

```sh
. /path/to/esp-idf/export.sh
idf.py set-target esp32s3
idf.py build
```

构建默认选择 Puppy 板、16 MB Flash 分区表 `partitions/16m.csv` 和安静陪玩模式。`CONFIG_PUPPY_STATIONARY_PLAY=y` 使电机控制与离线动作保持关闭。应用、表情资源和地址映射由构建生成，可在 `build/` 中查看。设备安装前请根据自己的板型、分区和现有设置选择部署方式。

首次接入网关时，在 `idf.py menuconfig` 的 `Puppy local companion WebSocket URL` 中填入设备可访问的网关地址，例如 `ws://puppy-gateway.local:8000/v1/audio`，并把 `Puppy local companion pairing token` 设为下面生成的设备令牌。默认令牌为空。已保存的有效网关覆盖配置优先使用。填好 URL 和令牌后再次执行 `idf.py build`，更新镜像。令牌写入本地构建配置后，固件也含该令牌，带真实令牌的镜像不要用于分发。

## macOS 本地网关

使用 Apple Silicon Mac、Python **3.11** 与系统 `libopus`（可通过 Homebrew 安装）。默认语音合成使用 macOS `say`。在候选源码根目录运行：

```sh
brew install python@3.11 opus
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r companion/requirements.txt
umask 077
cp gateway-config.example.json gateway-config.json
python - <<'PY'
from pathlib import Path
from secrets import token_urlsafe

for name in ('.device-token', '.parent-token'):
    with Path(name).open('x') as stream:
        stream.write(token_urlsafe(32) + '\n')
PY
```

将可用的本地 MLX 文本/视觉模型放到 `PUPPY_MODEL_PATH`，Whisper 模型放到 `models/whisper-small-mlx`。可从以下模型仓库按其条款取得文件。源码包附带固件资源，MLX 权重由使用者自行下载。

- [MLX Qwen3.5-4B-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit)
- [MLX Whisper Small](https://huggingface.co/mlx-community/whisper-small-mlx)

在已激活的网关 Python 环境中下载到示例默认目录。首次下载需要网络和足够的磁盘空间，之后启动器使用本地文件。

```sh
python -m pip install huggingface_hub
hf download mlx-community/Qwen3.5-4B-4bit --local-dir models/Qwen3.5-4B-4bit
hf download mlx-community/whisper-small-mlx --local-dir models/whisper-small-mlx
```

模型路径要包含完整的模型与处理器文件。可在 `gateway-config.json` 中更改 `PUPPY_MODEL_PATH`，Whisper 仍从网关根目录的 `models/whisper-small-mlx` 加载。

```sh
python -m companion.launch --root .
```

网关就绪后，在 Mac 上打开 `http://127.0.0.1:8000/`，用 `.parent-token` 的值登录家长网页。设备和网关需在可达的受信局域网内。`.device-token` 用于机身连接，与家长令牌独立。网关示例默认启用本地模型，额外语音识别、在线模型、Hermes 和远程同步均需按实际使用需求单独配置。

## 回归测试

使用只含候选源码的目录、测试数据库及合成夹具。以下命令所覆盖的测试不连接你的真实小狗：

```sh
python -m pip install -r companion/requirements-test.txt
python -m unittest discover -s companion -t .
python3 scripts/tests/test_puppy_safety.py
python3 scripts/tests/test_gateway_config.py
python3 scripts/tests/test_ssl_lifecycle.py
python3 scripts/tests/test_puppy_sleep_policy.py
node --test scripts/tests/test_memory_panel.cjs
```

固件宿主机测试需要启用 ESP-IDF 环境（`IDF_PATH`），并安装 Clang/Clang++。iOS 构建与测试见 [家长端说明](../parent-ios/README.md)。
