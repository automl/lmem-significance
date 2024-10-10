pytest --junitxml=tests/reports/junit/junit.xml --html=tests/reports/junit/report.html
genbadge tests -o tests/tests-badge.svg -i .\tests\reports\junit\junit.xml
coverage run  --data-file=./tests/reports/.coverage -m pytest tests/metafeature_test.py tests/sanity_test.py
coverage xml -o tests/reports/coverage.xml --data-file=./tests/reports/.coverage
coverage html -d tests/reports/cov_html --data-file=./tests/reports/.coverage
genbadge coverage -o tests/coverage-badge.svg -i .\tests\reports\coverage.xml
