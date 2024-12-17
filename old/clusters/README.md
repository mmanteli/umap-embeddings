# Clustering with von Mises-Fisher distribution

Attempt at von Mises-Fisher clustering. Unsuccessful due to package problems. I almost got it working though, so if you want to try, feel free! This has only been attempted on Mahti.

## Compatibility problems

``scikit-learn>0.2`` breaks ``spherecluster`` due to ``scikit.cluster.k_means_`` being renamed and moved under a new class. [Here](https://github.com/rfayat/spherecluster/tree/scikit_update) is a fork that claims to fix it, but I did not get it to work (possibly due to the python version problem as described below).


Python3.9 on Mahti failed to load older versions of ``scikit-learn<0.22`` and as the first "``spherecluster`` does not work!!" post was from early 2021, I moved to python3.6.

To set this up on Mahti:

```bash
python3.6 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The setup is **very volatile** and changing the versions will have an effect. E.g. ``numba==0.53.0`` (as opposed to ``0.51.0`` given in requirements) no longer works. In this setup, all package versions date to around Dec 2019 -- March 2020.
