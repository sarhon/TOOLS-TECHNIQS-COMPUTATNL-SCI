# Discrete Ordinates Transport Test Cases

This directory contains various test cases demonstrating different physical phenomena in neutron transport.

## Case Descriptions

### Case 1: Single Material with Reflective Boundaries
**File:** `1_material_ref/`

**Physics:** Single uniform material with reflective boundary conditions on both sides.
- Reflective boundaries trap neutrons in the system
- Higher flux buildup compared to vacuum boundaries
- Symmetric flux distribution

**Key Parameters:**
- Total XS: 1.0, Scatter: 0.9, Source: 1.0
- Boundaries: Reflective on both sides
- Domain: [-10, 10] cm

---

### Case 2: Two Materials with Vacuum Boundaries
**File:** `2_material_vac/`

**Physics:** Interface between two different materials showing flux discontinuity.
- Demonstrates flux matching at material interfaces
- Vacuum boundaries allow neutrons to escape
- Shows effect of different material properties

**Key Parameters:**
- Material 1: Total XS: 1.0, Scatter: 0.5
- Material 2: Total XS: 2.0, Scatter: 1.5
- Boundaries: Vacuum on both sides

---

### Case 3: Three Materials with Vacuum Boundaries
**File:** `3_material_vac/`

**Physics:** Complex multi-region problem with scatterers surrounding a source.
- Central source region (Q = 1.0)
- Flanked by high-scattering materials
- Demonstrates deep penetration and flux peaking

**Key Parameters:**
- Left/Right scatterers: Total XS: 2.0, Scatter: 1.99
- Central source: Total XS: 1.0, Scatter: 0.0, Q: 1.0
- Boundaries: Vacuum on both sides

---

### Case 4: Pure Absorption (No Scattering) ⭐ NEW
**File:** `4_pure_absorption/`

**Physics:** Material with ZERO scattering - only absorption.
- **Shows:** Exponential decay of flux
- No neutron redistribution in angle
- Clean demonstration of Beer's law attenuation
- All angular fluxes follow same exponential decay

**Key Parameters:**
- Total XS: 1.0, **Scatter: 0.0** (pure absorber)
- Uniform source throughout material
- Boundaries: Vacuum on both sides

**Expected Behavior:**
- Symmetric exponential decay from center
- All angular fluxes identical (no scattering to redistribute)
- Mean free path = 1/Σ_total = 1.0 cm

---

### Case 5: High Scattering ⭐ NEW
**File:** `5_high_scatter/`

**Physics:** Material with very high scattering-to-absorption ratio.
- **Shows:** Neutron diffusion behavior
- Scattering redistributes neutrons in angle and space
- Flux "spreads out" before being absorbed
- Compare to Case 4 to see scattering effect

**Key Parameters:**
- Total XS: 2.0, **Scatter: 1.98** (99% scattering!)
- Absorption: only 0.02 (1%)
- Uniform source throughout material

**Expected Behavior:**
- Much flatter flux distribution than pure absorber
- Angular fluxes show distinct directional dependence
- Neutrons travel further before absorption

---

### Case 6: Reflective Boundaries ⭐ NEW
**File:** `6_reflective_boundaries/`

**Physics:** Same material as Case 4, but with reflective boundaries.
- **Shows:** Effect of boundary conditions on flux
- Neutrons cannot escape → higher flux buildup
- Compare to vacuum boundary cases

**Key Parameters:**
- Total XS: 1.0, Scatter: 0.5
- **Boundaries: Reflective on BOTH sides**
- Uniform source

**Expected Behavior:**
- Much higher flux magnitudes than vacuum cases
- Symmetric flux distribution
- Flux gradient approaches zero at boundaries

---

### Case 7: Mixed Boundaries (Asymmetric) ⭐ NEW
**File:** `7_mixed_boundaries/`

**Physics:** Left reflective, right vacuum boundary.
- **Shows:** Asymmetric flux distribution
- Neutrons reflected on left, escape on right
- Interesting directional effects

**Key Parameters:**
- Total XS: 1.0, Scatter: 0.5
- **Left boundary: Reflective**
- **Right boundary: Vacuum**

**Expected Behavior:**
- Higher flux on left side (reflection)
- Lower flux on right side (escape)
- Asymmetric angular flux patterns

---

### Case 8: Void Penetration ⭐ NEW
**File:** `8_void_penetration/`

**Physics:** Neutron streaming through near-vacuum region.
- **Shows:** Deep penetration and streaming
- Source → Void → Absorber configuration
- Minimal interaction in void region

**Key Parameters:**
- Source region: Q = 10.0 (left side)
- **Void region: Total XS = 0.001** (nearly transparent)
- Absorber region: Total XS = 5.0 (right side)

**Expected Behavior:**
- Neutrons stream freely through void
- Sharp flux drop entering absorber
- Demonstrates ballistic transport in low-density regions

---

### Case 9: Dual Source ⭐ NEW
**File:** `9_dual_source/`

**Physics:** Two separate source regions with different strengths.
- **Shows:** Source superposition and interaction
- Weak source (left) vs strong source (right)
- Non-source absorber region in middle

**Key Parameters:**
- Left source: Q = 0.5 (weak)
- Middle: Q = 0.0 (no source, absorber only)
- Right source: Q = 2.0 (strong, 4× left)

**Expected Behavior:**
- Two flux peaks at source locations
- Flux dip in middle absorber region
- Asymmetric due to different source strengths

---

### Case 10: Absorption vs Scattering Comparison ⭐ NEW
**File:** `10_absorption_vs_scatter/`

**Physics:** Direct comparison of pure absorption vs high scattering.
- **Shows:** Side-by-side material behavior differences
- Pure absorber on left, high scatterer on right
- Source in middle feeding both

**Key Parameters:**
- Left: Pure absorber (Scatter: 0.0)
- Middle: Source (Q = 5.0)
- Right: High scatterer (Scatter: 1.9/2.0 = 95%)

**Expected Behavior:**
- Steep exponential decay into left (absorption)
- Gradual penetration into right (scattering redistributes)
- Asymmetric flux distribution demonstrates material differences

---

## Running the Cases

For each case, navigate to its directory and run both solvers:

```bash
cd cases/4_pure_absorption

# Run Fortran solver
../../fdiscord/bin/fdiscord -i input.json -o foutput.json

# Run Python solver
python3 -m pydiscord.cli -i input.json -o pyoutput.json

# Generate plots
python3 ../../postproc.py
```

## Comparison Matrix

| Case | Scattering | Sources | Boundaries | Key Physics |
|------|-----------|---------|------------|-------------|
| 1 | Medium | 1 uniform | Ref/Ref | Reflective BC |
| 2 | Medium | 1 uniform | Vac/Vac | Material interface |
| 3 | High | 1 central | Vac/Vac | Scattering barriers |
| **4** | **None** | **1 uniform** | **Vac/Vac** | **Pure absorption** |
| **5** | **Very high** | **1 uniform** | **Vac/Vac** | **Diffusion regime** |
| **6** | **Medium** | **1 uniform** | **Ref/Ref** | **Trapped neutrons** |
| **7** | **Medium** | **1 uniform** | **Ref/Vac** | **Asymmetric BC** |
| **8** | **Low** | **1 localized** | **Vac/Vac** | **Void streaming** |
| **9** | **Medium** | **2 separate** | **Vac/Vac** | **Multiple sources** |
| **10** | **Asymmetric** | **1 central** | **Vac/Vac** | **Material contrast** |

## Physical Insights

### Scattering Effects
Compare Cases 4 (no scatter) vs 5 (high scatter):
- Absorption → exponential decay
- Scattering → diffusion, flatter profiles

### Boundary Conditions
Compare Cases 4 (vac/vac) vs 6 (ref/ref) vs 7 (ref/vac):
- Vacuum → neutrons escape, lower flux
- Reflective → neutrons trapped, higher flux
- Mixed → asymmetric behavior

### Source Distribution
Compare Cases 3 (central source) vs 8 (edge source) vs 9 (dual sources):
- Central source → symmetric decay
- Edge source → one-sided decay
- Dual sources → superposition effects

### Material Properties
Case 10 directly shows:
- Pure absorption → steep gradients
- High scattering → gradual gradients
- Same total XS, different behavior!