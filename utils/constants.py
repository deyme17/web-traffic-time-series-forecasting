import re


# === data processing constants ===

LAG_DAYS = [7, 31, 91, 180, 365]

ACCESS = ["spider", "desktop", "mobile-web", "all-access"]
AGENTS = ["spider", "all-agents"]
LANGS = ["de", "en", "es", "fr", "ja", "ru", "zh", "other"]
SITES = ["wikipedia.org", "commons.wikimedia.org", "www.mediawiki.org"]

# page name format:  <article>_<lang>.wikipedia.org_<access>_<agent>
RE_PAGE = re.compile(
    r"^(.+?)_([a-z]{2,3})\.(?:(wikipedia\.org)|(commons\.wikimedia\.org)|(www\.mediawiki\.org))_"
    r"(all-access|desktop|mobile-web)_"
    r"(spider|all-agents)$"
)


# === feature size constants ====

N_LAGS = len(LAG_DAYS)
N_PAGE_CAT = len(ACCESS) + len(AGENTS) + len(LANGS) + len(SITES)
N_TEMPORAL = 4      # sin_dow, cos_dow, sin_month, cos_month
N_PAGE_SCALAR = 2   # year_autocorr, quarter_autocorr
N_PAGE_STAT = 3     # page_mean, page_std, page_vc


# === dataset/model constants

N_PAGE = N_PAGE_SCALAR + N_PAGE_STAT + N_PAGE_CAT
ENC_DIM = 1 + N_LAGS + N_LAGS + N_TEMPORAL + N_PAGE
DEC_DIM = ENC_DIM - 1
FINGERPRINT_SIGNAL = 1 + N_LAGS # fingerprint needs only hits and lags