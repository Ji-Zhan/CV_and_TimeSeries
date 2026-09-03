# Driver Fatigue Detection (司机疲劳实时监测系统)

基于 **NVIDIA Jetson Orin Nano** 边缘设备的实时司机疲劳监测系统，并扩展到**电池气体异常监测**。
采用 **YOLOv11n 眼/嘴分类器 + 关键点几何特征** 的混合融合方案 + 多信号滞回状态机，全链路 Docker Compose 端到端部署，零云端依赖。

> ⚠️ 本仓库只包含**源码、配置与文档**。模型权重 / 数据集 / 录像等大文件**不包含在内**，运行时从外部挂载或自行下载（见下文）。

## 架构（Docker Compose 五大容器）

```
摄像头 /dev/video0
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│ fatigue-detector (:8080)  PyTorch YOLO GPU FP16             │
│  · eye_mouth_detect  眼/嘴四分类 (逐帧)                    │
│  · yolo11n-pose      人脸5关键点 (每5帧)                   │
│  · 多信号融合 + 滞回状态机                                  │
│  · MJPEG 推流 /snapshot /status                            │
└──────────────┬──────────────────────────────────────────────┘
               │ HTTP POST /api/events (异步，失败本地缓存)
               ▼
┌──────────────────────────────────────────────────────────────┐
│ fatigue-api (:8000)  FastAPI + Dashboard                    │
└──────────────┬──────────────────────────────────────────────┘
               │ asyncpg
               ▼
┌──────────────────────────────────────────────────────────────┐
│ fatigue-db (:5432)  PostgreSQL 16  按日分表                 │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ gas-monitor (:8081)  电池气体 CSV 回放 + BiLSTM-AE 推理     │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ yolo11-jupyter (:8888)  JupyterLab 开发调试                 │
└──────────────────────────────────────────────────────────────┘
硬件告警：GPIO 蜂鸣器(疲劳) + 红/黄/绿 LED（烟/火/正常）
```

| 容器 | 端口 | 职责 |
|------|------|------|
| `fatigue-detector` | 8080 | YOLO 推理 + MJPEG 推流 |
| `fatigue-api` | 8000 | FastAPI + 网页 Dashboard |
| `fatigue-db` | 5432 | PostgreSQL 16（按日分表） |
| `gas-monitor` | 8081 | 电池气体 CSV 回放 + BiLSTM-AE 推理 |
| `yolo11-jupyter` | 8888 | JupyterLab 开发 |

## 疲劳检测核心管线（混合式）

**双模型互为备份**
- `eye_mouth_detect`（微调 YOLOv11n，逐帧）：4 类 `闭眼 / 张嘴 / 睁眼 / 闭嘴`
- `yolo11n-pose`（预训练，每 5 帧）：人脸 5 关键点 `[左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]`

**三个疲劳信号**
- 闭眼：分类器优先；漏检时用**瞳孔圆形度检测法**兜底
- 歪头/点头：5 点 `solvePnP(EPNP)` 解头部俯仰角，与**冻结基线**比较
- 打哈欠：嘴部纵横比 **MAR**（鼻尖近似上唇中点）

**判定**（多信号确认 + 滞回状态机）
- 单闭眼 / 单歪头 > 1.5s → 疲劳
- 闭眼+歪头 > 1.0s → 疲劳
- 闭眼+张嘴（哈欠）> 0.5s，**1 分钟内 ≥2 次** → 疲劳
- `正常 --疲劳帧≥60--▶ 疲劳  --正常帧≥30--▶ 正常`

## 电池气体异常监测

锂电池充电析出 H₂/CO + 温升异常预示**热失控**。用**仅以正常段训练的无监督 BiLSTM 自编码器**做时间序列异常检测，重建误差（MSE）作为异常分数，正常/事件段分数 ≥5× 分离验证，状态分级 Normal/Warning/Critical。

## 目录结构

```
tiring_detection/
├── backend/             FastAPI 后端 + Dashboard
│   ├── main.py          路由
│   ├── database.py      PostgreSQL 按日分表 (asyncpg)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── templates/dashboard.html
├── jetson_deploy/       边缘推理部署代码
│   ├── detector_service.py   主检测 + MJPEG
│   ├── fatigue_detector.py
│   ├── trt_inference.py
│   └── convert_models.sh
├── src/                 核心算法与训练/导出
│   ├── fatigue_state_machine.py  滞回状态机
│   ├── alert_handler.py          告警截图/片段
│   ├── event_logger.py           事件 JSON + HTTP 上报
│   ├── feature_extraction.py     特征计算
│   ├── test_drowsy_hybrid.py     单文件推理
│   └── train_*.py                训练/导出脚本
├── battery_gas/         电池气体监测
│   ├── gas_monitor.py           回放 + 推理
│   └── train_lstm_ae.py         BiLSTM-AE 训练
├── configs/             YOLO 配置
├── data/                （运行数据，未提交）
│   ├── raw_videos/  output_video/  clips/  screenshots/
│   ├── events/  gas_state  gas_uploads/  gas_model/
│   └── BEGINNER_GUIDE.md  TRAINING_HANDOFF.md
├── outputs/             （训练产物，未提交）
├── docker-compose.yml
└── .gitignore
```

## 运行

```bash
# 需要：NVIDIA Jetson（JetPack 6 / nvidia runtime）+ 外部模型与数据集挂载
docker compose -f tiring_detection/docker-compose.yml up -d
```

**访问入口**：Dashboard `:8000/api/dashboard` · 视频流 `:8080/stream` · API 文档 `:8000/docs` · JupyterLab `:8888`

## 外部依赖（本仓库不包含）

- 模型：`models/eye_mouth_detect/weights/best.pt`、`models/yolo11n-pose.pt`、`data/gas_model/*` → 从训练产出/预训练权重获取并挂载到 `/workspace/models`
- 数据集：NTHU-DDD（`data/driver-yolo11`、`data/driver-coco`）
- 容器基础镜像 `yolo11-jupyter` 由 Ultralytics Jetson 镜像构建
