autopep8 -i -a -a `find . | grep -i .py$`
isort -i `find . | grep -i .py$`
autoflake --in-place --remove-all-unused-imports `find . | grep -i .py$`
black `find . | grep -i .py$`
flake8 --ignore=E501
