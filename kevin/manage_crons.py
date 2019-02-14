from crontab import CronTab

# http://blog.appliedinformaticsinc.com/managing-cron-jobs-with-python-crontab/
cron = CronTab(user="minion")

print('Printing all cron tasks: ')
for job in cron:
    print(job)


print('Adding a new cron tasks: ')
job = cron.new(command='cd /var/www/kevin/kevin/ && /var/envs/kevin/bin/python action_aggregation.py --target=today',
               comment='minion_usasociety_aggregator')
# job.hour.every(6)

job.minute.on(0)
job.hour.during(0,23)

job.enable()
cron.write()
print('Finished crontab configuration with success.')