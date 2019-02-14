#!/bin/bash
ABS_REPO_PATH=$(unset CDPATH && cd "$(dirname "$0")/.." && echo $PWD)
cat <<EOF >${ABS_REPO_PATH}/.ci/blacklisted.json
{
    "sympy/physics/mechanics/tests/test_kane3.py": [
        "test_bicycle"
    ],
    "sympy/utilities/tests/test_wester.py": [
        "test_W25"
    ]
}
EOF
python3 -m pytest -ra --durations 0 --verbose | tee $ABS_REPO_PATH/.ci/durations.log
