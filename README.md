# mutation

**beta**

`mutation` check that tests are robust.

```sh
pip install mutation
mutation play tests.py --include="src/*.py"
mutation replay
```

Both `--include` and `--exclude` are optional but highly recommended
to avoid the production of useless mutations. `mutation` will only
mutate code that has test coverage, hence it works better with a high
coverage.

`mutation` will detect whether the tests can be run in parallel. It is
recommended to make the test suite work in parallel to speed up the
work of `mutation`.

Also, it is better to work with a random seed, otherwise add the
option `--randomly-seed=n` that works.

- [forge](https://git.sr.ht/~amirouche/mutation)
