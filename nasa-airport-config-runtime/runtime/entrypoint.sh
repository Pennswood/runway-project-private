#!/bin/bash
set -euxo pipefail
exit_code=0

{
    cd /codeexecution

    echo "Unpacking submission"
    unzip -n ./submission/submission.zip -d ./

    if [ -f "main.py" ]
    then
	echo "File list"
	find . -type f -exec sh -c 'printf "%s %s \n" "$(ls -l $1)" "$(md5sum $1)"' '' '{}' '{}' \;

	echo "List installed packages"
	conda list -n nasa-airport-config-runtime

	echo "Copying data from the cloud to local disk"
	cp -r /clouddata /data

	find /data -type d -exec chmod 700 {} \;
	find /data -type f -exec chmod 600 {} \;

	find /supervisor -type d -exec chmod 700 {} \;
	find /supervisor -type f -exec chmod 600 {} \;

	echo "Extracting test features"
	find /data -name '*.csv.bz2' -exec parallel -I% bunzip2 % ::: {} \+

	cp /data/submission_format.csv /codeexecution
	chmod 644 /codeexecution/submission_format.csv

	echo "Creating prediction time file"
	tail -n +2 /data/submission_format.csv | cut -d ',' -f2 | sort | uniq > /data/prediction_times.txt
	echo "Evaluating $(wc -l < /data/prediction_times.txt) time points"
	head /data/prediction_times.txt

	echo "Available disk"
	df -h /codeexecution

        echo "Running submission with Python"
	while read prediction_time
	do
	    conda run \
		  --no-capture-output \
		  -n nasa-airport-config-runtime \
		  python /supervisor/supervisor.py $prediction_time

	    find /extracts/$prediction_time -type d -exec chmod 755 {} \;
	    find /extracts/$prediction_time -type f -exec chmod 644 {} \;
	    ln -fns /extracts/$prediction_time data

	    sudo -u appuser \
		 /srv/conda/bin/conda run \
		 --no-capture-output \
		 -n nasa-airport-config-runtime \
		 python main.py $prediction_time

	    # Test that submission is valid
	    echo "Testing that prediction is valid"
	    conda run -n nasa-airport-config-runtime \
		  python /supervisor/scripts/check_prediction.py $prediction_time

	    mv /codeexecution/prediction.csv /predictions/${prediction_time}.csv

	done < /data/prediction_times.txt

	echo "Constructing submission from individual predictions"
	conda run -n nasa-airport-config-runtime python /supervisor/scripts/construct_submission.py

	echo "Testing that submission is valid"
	conda run -n nasa-airport-config-runtime pytest /supervisor/scripts/test_submission.py

	chown appuser:appuser submission/submission.csv

    else
        echo "ERROR: Could not find main.py in submission.zip"
        exit_code=1
    fi

    echo "================ END ================"
} |& tee "/codeexecution/submission/log.txt"

cp /codeexecution/submission/log.txt /tmp/log
exit $exit_code
