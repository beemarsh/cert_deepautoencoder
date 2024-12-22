import pandas as pd
import re

def preprocess_url(url_str):
    # Compute basic URL features
    url_len = len(url_str)
    url_depth = url_str.count('/') - 2

    # Extract domain name
    matches = re.findall("//(.*?)/", url_str)
    if matches:
        domainname = matches[0]
    else:
        # fallback if no trailing slash after domain
        domainname = url_str.split("//")[-1].split("/")[0]

    domainname = domainname.replace("www.", "")
    dn = domainname.split(".")
    # Simplify multi-level domains (e.g., sub.sub.domain.com -> domain.com)
    if len(dn) > 2 and not any(x in domainname for x in ["google.com", ".co.uk", ".co.nz", "live.com"]):
        domainname = ".".join(dn[-2:])

    # Categorize domain (1=other, 2=socnet, 3=cloud, 4=job, 5=leak, 6=hack)
    if domainname in ['dropbox.com', 'drive.google.com', 'mega.co.nz', 'account.live.com']:
        r = 3
    elif domainname in ['wikileaks.org', 'freedom.press', 'theintercept.com']:
        r = 5
    elif domainname in [
        'facebook.com','twitter.com','plus.google.com','instagr.am','instagram.com',
        'flickr.com','linkedin.com','reddit.com','about.com','youtube.com','pinterest.com',
        'tumblr.com','quora.com','vine.co','match.com','t.co'
    ]:
        r = 2
    elif domainname in ['indeed.com','monster.com','careerbuilder.com','simplyhired.com']:
        r = 4
    elif ('job' in domainname and ('hunt' in domainname or 'search' in domainname)) \
         or ('aol.com' in domainname and ("recruit" in url_str or "job" in url_str)):
        r = 4
    elif domainname in [
        'webwatchernow.com','actionalert.com','relytec.com','refog.com','wellresearchedreviews.com',
        'softactivity.com','spectorsoft.com','best-spy-soft.com'
    ]:
        r = 6
    elif 'keylog' in domainname:
        r = 6
    else:
        r = 1

    return [r, url_len, url_depth]

# Load the CSV (ensure the file has a column named "url")
df = pd.read_csv('http_final_red_renamed.csv', dtype=str)

# Apply the function to extract URL-based features
df[['domain_category', 'url_len', 'url_depth']] = df.apply(
    lambda row: preprocess_url(row['url']),  # <-- Use column "url"
    axis=1,
    result_type='expand'
)

# Save the processed DataFrame
df.to_csv('http_url.csv', index=False)

print("URL preprocessing complete. 'domain_cat', 'url_len', and 'url_depth' have been added to 'http_preprocessed_url.csv'.")
