from datetime import datetime, date, timedelta
from dateutil.relativedelta import *
from dateutil.parser import *

date_string = [
    "i ate food 3 days ago",
    "i ate food two month ago",
    "i ate food 1 year ago",
    "i ate food next week",
    "i ate food last month",
    "i ate food day before yesterday",
    "i ate food tomorrow",
    "i ate food on 28-03-2023",
]

def parse_date_string(date_string):
    today = datetime.today().date()
    if 'today' in date_string:
        return today
    elif 'day before yesterday' in date_string:
        return today - relativedelta(days=2)
    elif 'day after tomorrow' in date_string:
        return today + relativedelta(days=2)
    elif 'yesterday' in date_string:
        return today - relativedelta(days=1)
    elif 'tomorrow' in date_string:
        return today + relativedelta(days=1)
    elif 'last' in date_string:
        last = parse(date_string, fuzzy=True)
        return last.date()
    elif 'ago' in date_string:
        n_days_ago = int(date_string.split()[0])
        return today - relativedelta(days=n_days_ago)
    else:
        return today

def determine_date(allusion):
    today = date.today()
    parsed_date = parse(allusion, fuzzy=True)
    print(f'parsed_date: {parsed_date}')

    if parsed_date.date() == today:
        print(f'parsed_date.date() == today:')
        return today

    if parsed_date.date() > today:
        delta = relativedelta(parsed_date.date(), today)
        print(f'parsed_date.date() > today: {today} - {delta}')
        return today - delta

    if parsed_date.date() < today:
        delta = relativedelta(today, parsed_date.date())
        print(f'parsed_date.date() < today: {today} + {delta}')
        return today + delta

    if "yesterday" in allusion:
        print(f'yesterday')
        return today - timedelta(days=1)

    if "tomorrow" in allusion:
        print(f'tomorrow')
        return today + timedelta(days=1)

    if "day before yesterday" in allusion:
        print(f'day before yesterday')
        return today - timedelta(days=2)

    if "day after tomorrow" in allusion:
        print(f'day after tomorrow')
        return today + timedelta(days=2)

    if "last" in allusion:
        print(f'last')
        if "week" in allusion:
            print(f'week')
            return parsed_date.date() - timedelta(days=parsed_date.weekday() + 7)
        if "month" in allusion:
            print(f'month')
            return parsed_date.date().replace(day=1) - relativedelta(days=1)
        if "year" in allusion:
            print(f'year')
            return parsed_date.date().replace(month=1, day=1) - relativedelta(days=1)

    if "next" in allusion:
        print(f'next')
        if "week" in allusion:
            print(f'week')
            return parsed_date.date() + timedelta(days=7 - parsed_date.weekday())
        if "month" in allusion:
            print(f'month')
            return parsed_date.date().replace(day=28) + relativedelta(days=4)
        if "year" in allusion:
            print(f'year')
            return parsed_date.date().replace(month=12, day=31) + relativedelta(days=1)

    if "ago" in allusion:
        print(f'ago')
        if "days" in allusion:
            print(f'days')
            days = int(allusion.split()[0])
            return today - timedelta(days=days)

        if "weeks" in allusion:
            print(f'weeks')
            weeks = int(allusion.split()[0])
            return today - timedelta(weeks=weeks)

        if "months" in allusion:
            print(f'months')
            months = int(allusion.split()[0])
            return today.replace(day=1) - relativedelta(months=months)

        if "years" in allusion:
            print(f'years')
            years = int(allusion.split()[0])
            return today.replace(month=1, day=1) - relativedelta(years=years)

    # Default case: return None if the string is not a valid allusion to a date
    return None

for string in date_string:
    date_value = determine_date(string).strftime('%d-%m-%Y')
    print(date_value)  # Output: 2022-04-22