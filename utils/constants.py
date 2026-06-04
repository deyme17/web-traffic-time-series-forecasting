import re


# === data processing constants ===


LAG_DAYS = [7, 31, 180, 365]

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

MAX_GAP_INTERPOLATE = 7     # if gaps <= MAX_GAP_INTERPOLATE -> linear interpolation
NAN_DROP_THRESHOLD = 0.3    # drop page if fraction of NaNs > NAN_DROP_THRESHOLD
WINSOR_K = 4.               # spike threshold: median +- WINSOR_K * MAD