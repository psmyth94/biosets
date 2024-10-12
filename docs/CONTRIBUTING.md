# Contributing to BioSets

## How to Create a Pull Request?

If you want to contribute to the codebase, follow these steps:

1. **Set up SSH Key:**

   First, set up an SSH key in GitHub by running:

   ```bash
   # Generate a new SSH key (replace with your actual email)
   ssh-keygen -t rsa -b 4096 -C "your-email@email.com"

   # Start the SSH agent
   eval "$(ssh-agent -s)"

   # Add your SSH key to the ssh-agent
   ssh-add ~/.ssh/id_rsa

   # Copy the SSH key to your clipboard (Linux)
   cat ~/.ssh/id_rsa.pub | xclip -selection clipboard

   # On macOS
   pbcopy < ~/.ssh/id_rsa.pub

   # On Windows
   cat ~/.ssh/id_rsa.pub | clip
   ```

   Then add the SSH key to your GitHub account by going to your profile settings and
   clicking on the `SSH Keys` tab. Click on the `Add SSH Key` button and paste the key
   into the `Key` field.

2. **Fork the Repository:**

   Fork the repository on GitHub by clicking on the `Fork` button in the top right corner
   of the repository page.

3. **Clone the Forked Repository:**

   Clone the forked repository to your local machine:

   ```bash
   git clone git@github.com:your-username/biosets.git
   ```

3. **Create a New Branch:**

   Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   Do not work on the `main` branch directly. Always create a new branch for your
   changes.

4. **Set Up a Development Environment:**

   Set up a development environment by running the following command:

   ```bash
   conda env create -n biosets-local python=3.10 -c conda-forge
   conda activate biosets-local
   pip install -e ".[test]"
   ```

   (If BioSets was already installed in the virtual environment, remove it with
   `pip uninstall biosets` before reinstalling it in editable mode with the `-e` flag.)

5. **Develop the Features on Your Branch:**

   Make your changes to the code.

6. **Format Your Code:**

   Format your code. Run `ruff` so that your newly added files look nice with the
   following command:

   ```bash
   ruff .
   ```

7. **(Optional) Use Pre-commit Hooks:**

   You can also use pre-commit to format your code automatically each time you run
   `git commit`, instead of running `ruff` manually. To do this, install pre-commit via
   `pip install pre-commit` and then run `pre-commit install` in the project's root
   directory to set up the hooks. Note that if any files were formatted by pre-commit
   hooks during committing, you have to run `git commit` again.

8. **Commit Your Changes:**

   Once you're happy with your contribution, add your changed files and make a commit
   to record your changes locally:

   ```bash
   git add -u
   git commit -m "Your commit message"
   ```

9. **Sync with the Original Repository:**

   It is a good idea to sync your copy of the code with the original repository
   regularly. This way you can quickly account for changes:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

10. **Push the Changes:**

    Once you are satisfied, push the changes to the remote repository using:

    ```bash
    git push origin a-descriptive-name-for-my-changes
    ```

11. **Create a Pull Request:**

    Go to the webpage of the repository on GitHub. Click on "Pull request" to send
    your changes to the project maintainers for review.

## Code of Conduct

This project adheres to the Contributor Covenant code of conduct. By participating, you
are expected to abide by this code.
