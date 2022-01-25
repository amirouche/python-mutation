# mutation

**beta**

`mutation` check that tests are robust.

## Getting started

```sh
pip install mutation
mutation play tests.py --include="src/*.py"
mutation replay
```

## Usage

```
Usage:
  mutation play [--verbose] [--exclude=<globs>] [--only-deadcode-detection] [--include=<globs>] [--sampling=<s>] [--randomly-seed=<n>] [--max-workers=<n>] [<file-or-directory> ...] [-- TEST-COMMAND ...]
  mutation replay [--verbose] [--max-workers=<n>]
  mutation list
  mutation show MUTATION
  mutation apply MUTATION
  mutation (-h | --help)
  mutation --version
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

## TODO

- [ ] Avoid mutations that are syntax errors to improve efficiency, requires python 3.9;
- [ ] Add PyPy support in continuous integration;

## Links

- [forge](https://git.sr.ht/~amirouche/mutation)
