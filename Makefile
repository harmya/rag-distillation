COMMIT_MESSAGE := "Added new code, copied files to ssh server"

.PHONY: all clean

all: commit

commit:
	@echo "Adding changes..."
	git add src/*.py
	@echo "Committing changes..."
	git commit -m $(COMMIT_MESSAGE)
	@echo "Pushing changes..."
	git push

clean:
	@echo "Cleaning up..."