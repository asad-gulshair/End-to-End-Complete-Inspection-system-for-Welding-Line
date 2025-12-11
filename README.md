AI Inspection System
Weld Defect Detection • Coating Defect Detection • Flange Dimension Compliance (ASME Based)

A full end-to-end Artificial Intelligence inspection platform capable of analyzing weld surfaces, coating conditions, and flange dimensions based on industrial standards.
This repository includes the complete backend, frontend, sample input/output media, and demo video — except trained model files.

🚀 Overview

 AI Inspection System provides three powerful AI modules in one platform:

🔧 1. Weld Defect Detection

YOLO-based inspection

Detects cracks, porosity, undercut, slag, inclusions, lack of fusion, burn-through, etc.

Supports image & video processing

High-accuracy bounding boxes + confidence scores

🎨 2. Coating Defect Detection

Detects coating cracks, blistering, corrosion, peeling, holidays, and surface damage

Works with high-resolution input images

AI identifies defect regions automatically

🔩 3. Flange Dimension Compliance System (ASME / Applicable Standard Based)

A complete dimension-verification and compliance-testing system for industrial flanges.

The system:

✔ Measures the flange using:

ArUco markers

Depth estimation

Camera calibration

PnP pose estimation

Pixel-to-mm / inch conversion

Geometric analysis

✔ Extracts key flange dimensions:

Outer diameter

Inner diameter

Bolt circle diameter

Thickness

Number of bolt holes

Pitch

Hole diameter

Raised face dimensions

Other critical flange parameters

✔ Compares measurements with ASME flange standards

The backend uses:

data/flange_specifications11.csv


This file contains ASME or applicable industry standard dimensions.

✔ PASS / FAIL decision

The system automatically determines compliance:

PASS → flange dimensions are within standard tolerance

FAIL → flange is out of ASME limits

This makes the system fully ready for industrial QA/QC inspection, eliminating manual measurement errors.DATA IS only AVAIBLABE for research purpose.
