# Clustering with von Mises-Fisher distribution

## Compatibility problems

``scikit-learn>0.2`` breaks ``spherecluster`` due to ``scikit.cluster.k_means_``. [Here](https://github.com/rfayat/spherecluster/tree/scikit_update) is an update that claims to fix it, but I did not get it to work (possibly the same version problem as described below).


Python3.9 on Mahti failed to load older versions (than ``<0.22``) of ``scikit-learn`` and as the first "``spherecluster`` does not work!!" post was from 2021, I moved to python3.6 (in use in 2020) which is not default on Mahti! On python3.6:

```bash
python3.6 -m venv cenv
source cenv/bin/activate
pip install -r requirements.txt
pip install spherecluster
```
where ``requirements.txt`` contained:
```
numpy
scipy
scikit-learn<=0.22
pytest
nose
joblib
```

After this:
```bash
python
```

```python
from spherecluster import VonMisesFisherMixture
```
results in warnings but not an error as previously.