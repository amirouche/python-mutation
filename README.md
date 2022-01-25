# 🐛 mutation 🐛

**early draft** requires more testing, please report any findings in
[my public inbox](https://lists.sr.ht/~amirouche/public-inbox).

The goal of `mutation` is to give an idea on how robust, and help
improve test suites.

## Getting started

```sh
pip install mutation
mutation play tests.py --include="foobar/ex.py,foobar/__init__.py" --exclude="tests.py"
```

Then call:

```sh
mutation replay
```

## Usage

```
mutation play [--verbose] [--exclude=<globs>] [--only-deadcode-detection] [--include=<globs>] [--sampling=<s>] [--randomly-seed=<n>] [--max-workers=<n>] [<file-or-directory> ...] [-- TEST-COMMAND ...]
mutation replay [--verbose] [--max-workers=<n>]
mutation list
mutation show MUTATION
mutation apply MUTATION
mutation (-h | --help)
mutation --version
```

Both `--include` and `--exclude` support glob patterns. They are
optional but highly recommended to avoid the production of useless
mutations. 

`mutation` will only mutate code that has test coverage, hence it
works better with a high coverage.

`mutation` will detect whether the tests can be run in parallel. It is
recommended to make the test suite work in parallel to speed up the
work of `mutation`.

Also, it is better to work with a random seed, otherwise add the
option `--randomly-seed=n` that works.

## TODO

- [ ] Avoid mutations that are syntax errors to improve efficiency,
      use python 3.9+ `ast.unparse`, possibly with `black`;
- [ ] `mutation play` can error even if the code and the test suite is
      valid e.g. removing a docstring will trigger an error: reduce,
      and hopefully eliminate that problem, requires python 3.9+ (see
      above);
- [ ] Add PyPy support;

## Links

- [forge](https://git.sr.ht/~amirouche/mutation)
