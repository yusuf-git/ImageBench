Echo off
cls
echo activating the ImageBench v2 environment...
call conda activate imagebench

c:
cd\
cd py\imagebench\v2

echo starting ImageBench v2...
call python imagebench_driver.py -j img-comp-args.json
echo finished the algo operation
call conda deactivate
echo on

