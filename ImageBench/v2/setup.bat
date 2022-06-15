echo off
cls

call conda deactivate
REM if "%~1"=="y"   (
   REM echo "removing the ImageBench environment..."
   REM   call conda env remove --name ImageBench 
   REM   echo "done with the removal"
   REM   cls)

REM if "%~1"=="Y"  (echo "deleting the environment-1" call conda env remove --name ImageBench)


echo creating python, computer vision and ImageBench envs...
echo

call conda env create -f ImageBench_v2.yml
md C:\py\ImageBench\v2
md C:\py\ImageBench\v2\workspace
md C:\py\ImageBench\v2\BRISK_FLANN_baselines

echo copying ImageBench files...
copy *.py C:\py\ImageBench\v2
copy *.json C:\py\ImageBench\v2
copy *.bat C:\py\ImageBench\v2
xcopy sampling C:\py\ImageBench\v2\sampling /E /S /I
echo
cls
echo activating the ImageBench environment...

call conda activate imagebench
c:
cd\
cd py\imagebench\v2

cls
echo ImageBench v2 - setup done
call conda deactivate
echo on