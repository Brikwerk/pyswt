# pyswt
A Python implementation of the Stroke Width Transform

## Installation
Virtualenv is recommened for managing python packages.

Install the packages used in this project with:

```bash
pip install -r requirements.txt
```

To test with example code, run *test_pyswt.py* located at the repo root.

To use in your own project:

```python
import pyswt
img = cv2.imread(img_path)
bboxes = pyswt.run(img) # Outputs bounding boxes of found text
```