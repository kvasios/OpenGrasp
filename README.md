# OpenGrasp

**Apache-2.0 licensed Vision-to-Grasp engine for professional robot assistants and autonomous laboratories.**

OpenGrasp provides high-speed 6-DoF grasp pose prediction for unknown objects, designed for production integration without restrictive licensing or per-site engineering.

---

## Current Status

**TRL 4.** Functional laboratory pipeline operational on a Franka Research 3 arm with parallel-jaw gripper and Intel RealSense RGB-D sensing. The software scaffold, evaluation protocol, and data-generation pipeline are defined.

Active development is underway toward a packaged, benchmarked v1.0 release — including industrial feasibility filtering and towards adaptive grasp learning.

---

## Hardware

Developed and validated on:
- Franka Research 3 collaborative robot arm
- Weiss WSG-50 parallel-jaw gripper
- Intel RealSense D-series RGB-D cameras
- NVIDIA Jetson Orin Nano (edge inference target)

---

## License

Licensed under the [Apache License 2.0](LICENSE). Weights trained exclusively on synthetic and project-generated data — clean for commercial integration.

---

## Citation

If you use OpenGrasp in your research, please cite this repository. A technical paper is in preparation.