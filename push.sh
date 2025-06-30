@echo off
echo Adding files...
git add *

echo Committing...
git commit -m "."

echo Pushing to origin main...
git push origin main

echo Done.
pause
