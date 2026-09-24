# 人体电磁代理参数与来源（2026-09-23）

[文档索引](README.md) · [当前 Sionna 入口](sionna-import.md)。本页记录参数来源，不表示人体电磁模型已经过测量校准。

## 已核对来源

- S. Gabriel, R. W. Lau, C. Gabriel (1996). The dielectric properties of biological tissues: III. Parametric models for the dielectric spectrum of tissues. Physics in Medicine & Biology 41, 2271–2293. DOI: https://doi.org/10.1088/0031-9155/41/11/003 。书目信息经 IFAC-CNR References 页交叉核对。
- D. Andreuccetti, R. Fossi, C. Petrucci, IFAC-CNR tissue-property calculator (1997–2024), https://webnir.eu/04-Dosimetry/01-webapp.php?lang=gb 。旧 niremf.ifac.cnr.it 地址当前重定向至此；基于 Gabriel 参数模型，适用范围 10 Hz–100 GHz。
- 数据响应列由公开 tissprop.js 确认：name, frequency_Hz, conductivity_S_m, relative_permittivity, loss_tangent, wavelength_m, penetration_depth_m。

对 Muscle、3.5 GHz 的实际查询返回：εr = 51.444229951809085，σ = 2.557518249544951 S/m。原 σ=2.16 无法由该频点核实，已替换。结果是参数模型计算值，不是本项目测量值。

复现公开查询（只做参数计算）：

    curl --noproxy '*' https://webnir.eu/pyapp/gw2py.py --data 'modulopy=tissprop&funzoper=calcola&func=stsf&tiss=Muscle&freq=3500000000'

精确响应保存于 artifacts/review_20260923/sources/tissue-muscle.json；网页、JS 与书目页面同时保存并列出 SHA256。

## 仍是建模假设的部分

全身 SMPL 外表面用均匀 muscle 材质和 0.02 m 等效 slab，是工程代理；没有声称皮肤、脂肪、骨骼和衣物都等于肌肉。厚度并非从这篇文献推导。现 CLI 只允许 3.5 GHz，避免把单频常数静默外推。

Sionna 的薄层 Fresnel 模型、封闭人体网格和多层真实人体不同。当前验证证明几何改变信道，尚不能证明真实人体衰减或散射精度；正式研究需材质/厚度敏感性、人体测量或更精细模型对照。
