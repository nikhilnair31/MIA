# linkedIn_Job_Scraper

## General
* Idea is to somehow scrape LinkedIn job listings on a timed interval

## Run the following to Deploy
* Login first with 
    * `aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 832214191436.dkr.ecr.ap-south-1.amazonaws.com`
* One time 
    * `aws ecr create-repository --repository-name linkedinjobscraper`

* Run on every update
    * Build docker image with `docker build -t linkedinjobscraper .`
    * Add latest tag with `docker tag linkedinjobscraper:latest 832214191436.dkr.ecr.ap-south-1.amazonaws.com/linkedinjobscraper:latest`
    * Push to AWS ECR with `docker push 832214191436.dkr.ecr.ap-south-1.amazonaws.com/linkedinjobscraper:latest`

## How To Use
* Create a `.env` file containing the relevant variables from the `scraper.py` script
* Run the `scraper.py` script
* Refer to CSV data saved