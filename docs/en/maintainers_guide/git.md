# Git / Github Guide

## Git

* Official Git documentation: <https://git-scm.com/doc>

* Keep your commits clean! **Do not commit any unnecessary files!**
  Always type `git status` to check your commit before pushing.
  And modify `.gitignore` file if necessary.

* **Do not force push on the main branch!**

* Keep git history clean. Avoid unnecessary merge commits.
  Get familiar with the following commands:
```bash
git reset --soft/--hard HEAD/HEAD~/HEAD~2/<commit-id>/...
git stash
git stash pop
git rebase base_branch local_branch
```

* Useful git commands:
```bash
git config list (--global/--system/--local)
git config --global user.name "name"
git config --global user.email "aaa@aa.com"
git config --local --unset user.email

git remote -v
git remote add upstream https://xxx.git
git remote remove/rename
git remote update origin --prune

git checkout -b new_branch upstream/branch

git branch
git branch -a
git branch -d/-D name

git push upstream -u local_branch:remote_branch

git add .
git status
git restore --staged <file>
git commit -m "message"
git reset --soft/--hard HEAD^

git stash
git stash pop

git fetch
git pull  # suggest to set fast-forward only
git push

git tag <tagname>
git tag list
git push --tags

git rebase
```


## Github

### Reference a GitHub issue in commit message

To reference a GitHub issue in commit message, use `#issue_number`. For example:
```bash
git commit -m "Fix XXX (#123)"
```

This commit will be automatically linked to issue #123 on GitHub,
and a message will be posted on the issue page.
