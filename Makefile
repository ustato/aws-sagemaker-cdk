.DEFAULT_GOAL := help

.PHONY: help
help: ## show the help menu
	@grep -E '^[a-zA-Z][a-zA-Z._-]*:.*?## .*$$' $(MAKEFILE_LIST) \
		| sed -e 's/.*Makefile://g' \
		| awk 'BEGIN {FS = ": ## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

POETRY := poetry run
