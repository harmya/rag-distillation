# Define variables
COMMIT_MESSAGE := "Automatic commit"

# Define targets
.PHONY: all clean

all: commit

commit:
	@echo "Adding changes..."
	git add *
	@echo "Committing changes..."
	git commit -m $(COMMIT_MESSAGE)
	@echo "Pushing changes..."
	git push

clean:
	@echo "Cleaning up..."
	# Add cleanup commands if needed