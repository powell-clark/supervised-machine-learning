---
name: Lesson Completion Task
about: Track completion of stub lessons (4-8)
title: 'Complete Lesson [NUMBER]: [ALGORITHM NAME]'
labels: ['lesson-development', 'priority-high', 'content']
assignees: ''
---

## Lesson Information

**Lesson**: [e.g., 4a - SVM Theory]
**Algorithm**: [e.g., Support Vector Machines]
**Current Status**: 🚧 Stub (XXX lines)
**Target Status**: ✅ Complete (1,000-1,500 lines)

## Progress Tracking

### Content Sections

- [ ] **Introduction** (100-200 lines)
  - [ ] Story-driven motivation
  - [ ] Learning objectives
  - [ ] Table of contents with working links

- [ ] **Mathematical Foundation** (600-800 lines)
  - [ ] Complete derivations from first principles
  - [ ] Worked numerical examples
  - [ ] Geometric visualizations
  - [ ] Complexity analysis

- [ ] **From-Scratch Implementation** (400-600 lines)
  - [ ] Complete class with comprehensive docstrings
  - [ ] Type hints on all methods
  - [ ] Detailed inline comments
  - [ ] Step-by-step walkthrough

- [ ] **Real-World Application** (500-700 lines)
  - [ ] Dataset loading and EDA
  - [ ] Training with progress tracking
  - [ ] Comprehensive evaluation
  - [ ] Error analysis

- [ ] **Visualizations** (300-400 lines total)
  - [ ] Algorithm behavior
  - [ ] Decision boundaries
  - [ ] Training convergence
  - [ ] Hyperparameter effects

- [ ] **When to Use Guide** (200-300 lines)
  - [ ] 5+ ideal use cases
  - [ ] 5+ anti-use cases
  - [ ] Comparison with alternatives

- [ ] **Conclusion** (50-100 lines)
  - [ ] Key takeaways
  - [ ] Preview of next lesson
  - [ ] Further reading

### Quality Checks

- [ ] **Runs in Google Colab** (verified, no errors, < 5 min)
- [ ] **Passes "Lesson 1a Test"** (similar quality to gold standard)
- [ ] **All checklist items from LESSON_QUALITY_CHECKLIST.md** ✅
- [ ] **Line count target met** (1,000+ for theory, 800+ for practical)
- [ ] **Cell count target met** (45-60 for theory, 40-50 for practical)
- [ ] **Minimum visualizations** (8+ for theory, 12+ for practical)

## Development Resources

**Reference Documents**:
- 📋 [Content Restoration Plan](../../CONTENT_RESTORATION_PLAN.md)
- ✅ [Quality Checklist](../../LESSON_QUALITY_CHECKLIST.md)
- 🚀 [Development Guide](../../DEVELOPMENT_GUIDE.md)
- 📊 [Review Document](../../REVIEW_2025-2026.md)

**Gold Standard Examples**:
- Lesson 1a: Logistic Regression Theory (2,997 lines)
- Lesson 2a: Decision Trees Theory (3,385 lines)
- Lesson 2b: Decision Trees Practical (5,941 lines)

**Template**:
- [Starter Template](../../notebooks/4a_svm_theory_TEMPLATE.ipynb) (if available)

## Estimated Effort

**Time**: 10-12 hours focused development
**Timeline**: 2 weeks (part-time) or 1 week (full-time)

**Breakdown**:
- Days 1-2: Introduction + Math derivations (600 lines)
- Days 3-4: From-scratch implementation (400 lines)
- Days 5-6: Real-world application (500 lines)
- Day 7: Polish, test, review

## Definition of Done

✅ All content sections complete
✅ All quality checks passed
✅ Runs without errors in fresh Colab session
✅ Reviewed against Lesson 1a/2a quality
✅ All checklist items checked
✅ PR submitted and approved

## Additional Notes

<!-- Add any specific notes, challenges, or questions here -->
