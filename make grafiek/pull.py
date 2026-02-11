import os, time, json, requests, urllib3
from pathlib import Path

BASE_URL = "https://195.240.122.183"
BASE_URL2 = "http://192.168.1.181"
DOWNLOAD_DIR = Path(r"./datalogs")
STATE_FILE = DOWNLOAD_DIR / "state.json"
USERNAME = "Web User"
PASSWORD = "KeL@urens"
COOKIE_FILE = Path("./195.240.122.183_cookies.txt")
POLL_INTERVAL_SECONDS = 60 * 60

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "nl-NL,nl;q=0.9",
    "Connection": "keep-alive",
    "Referer": BASE_URL + "/logo_log_list.html"
}

def ensure_dirs():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps({}, indent=2))

def load_state():
    return json.loads(STATE_FILE.read_text())

def save_state(st):
    STATE_FILE.write_text(json.dumps(st, indent=2))

def parse_datalog_index(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    if lines[0].isdigit():
        return lines[1:]
    return lines

def load_cookies_from_netscape(sess: requests.Session, txt_path: Path):
    if not txt_path.exists():
        return False
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                parts = line.strip().split()
            if len(parts) < 7:
                continue
            domain, flag, path, secure, expiry, name, value = parts[:7]
            sess.cookies.set(name, value, domain=domain, path=path)
    return True

def save_cookies_to_netscape(sess: requests.Session, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for c in sess.cookies:
            f.write("\t".join([
                c.domain or "195.240.122.183",
                "FALSE",
                c.path or "/",
                "TRUE" if c.secure else "FALSE",
                "0",
                c.name,
                c.value
            ]) + "\n")

def get_index_requests(sess):
    sess.headers.update(BROWSER_HEADERS)
    r = sess.get(BASE_URL + "/lsc/datalogs", timeout=15, verify=False)
    if r.status_code != 200:
        raise RuntimeError(f"Index returned {r.status_code}")
    return parse_datalog_index(r.text)

def download_csv_requests(sess, filename):
    url = BASE_URL + f"/lsc/datalogs/{filename}"
    r = sess.get(url, timeout=20, verify=False)
    if r.status_code != 200:
        raise RuntimeError(f"Download {filename} returned {r.status_code}")
    return r.text

def append_new_lines(local_path: Path, server_text: str, state, filename):
    remote_lines = [l for l in server_text.splitlines() if l.strip()]
    if not remote_lines:
        return 0
    if not local_path.exists():
        local_path.write_text("")
    with local_path.open("r", encoding="utf-8", errors="ignore") as f:
        local_lines = [l for l in f.read().splitlines() if l.strip()]
    local_count = len(local_lines)
    if local_count >= 20000:
        print(f"{filename}: lokaal al >=20000 lines, overslaan.")
        state.setdefault(filename, {})["lines"] = local_count
        return 0
    added = 0
    if len(remote_lines) > local_count and remote_lines[:local_count] == local_lines[:local_count]:
        to_add = remote_lines[local_count:]
        allowed = 20000 - local_count
        to_add = to_add[:allowed]
        if to_add:
            with local_path.open("a", encoding="utf-8", errors="ignore") as f:
                f.write("\n".join(to_add) + "\n")
            added = len(to_add)
    else:
        if local_count > 0:
            last_local = local_lines[-1]
            try:
                idx = remote_lines.index(last_local)
            except ValueError:
                idx = -1
            if idx >= 0 and idx+1 < len(remote_lines):
                to_add = remote_lines[idx+1:]
                allowed = 20000 - local_count
                to_add = to_add[:allowed]
                with local_path.open("a", encoding="utf-8", errors="ignore") as f:
                    f.write("\n".join(to_add) + "\n")
                added = len(to_add)
            else:
                if len(remote_lines) > local_count and local_count < 20000:
                    to_add = remote_lines[local_count:20000]
                    with local_path.open("a", encoding="utf-8", errors="ignore") as f:
                        f.write("\n".join(to_add) + "\n")
                    added = len(to_add)
        else:
            to_add = remote_lines[:20000]
            local_path.write_text("\n".join(to_add) + "\n")
            added = len(to_add)
    new_local_count = len([l for l in local_path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()])
    state.setdefault(filename, {})["lines"] = new_local_count
    return added

def get_session_via_selenium_and_save_cookie():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    opts = Options()
    opts.add_argument('--ignore-certificate-errors')
    opts.add_argument('--allow-insecure-localhost')
    opts.add_argument('--ignore-ssl-errors=yes')
    opts.headless = True
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(30)
    driver.get(BASE_URL + "/logo_login.html")
    wait = WebDriverWait(driver, 10)
    try:
        user_select = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "select[name='name'], select#name, select[name='user']")))
    except:
        user_select = None
    if user_select:
        for opt in user_select.find_elements(By.TAG_NAME, "option"):
            if USERNAME in opt.text:
                opt.click()
                break
    try:
        keepsignin = driver.find_element(By.ID, "check_keepsignin")
        if not keepsignin.is_selected():
            keepsignin.click()
    except:
        pass
    pwd = driver.find_element(By.ID, "input_password")
    pwd.clear()
    pwd.send_keys(PASSWORD)
    driver.find_element(By.ID, "button_login").click()
    time.sleep(2)
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    for c in driver.get_cookies():
        s.cookies.set(c['name'], c['value'], domain=c.get('domain'), path=c.get('path'))
    driver.quit()
    save_cookies_to_netscape(s, COOKIE_FILE)
    return s

def run_once():
    ensure_dirs()
    st = load_state()
    sess = requests.Session()
    sess.headers.update(BROWSER_HEADERS)
    cookies_loaded = False
    if COOKIE_FILE.exists():
        cookies_loaded = load_cookies_from_netscape(sess, COOKIE_FILE)
        if cookies_loaded:
            print("Cookies geladen uit", COOKIE_FILE)
    if not cookies_loaded:
        print("Geen geldige cookies, inloggen via Selenium...")
        sess = get_session_via_selenium_and_save_cookie()
    try:
        files = get_index_requests(sess)
        if not files:
            print("Geen bestanden gevonden, opnieuw inloggen via Selenium...")
            sess = get_session_via_selenium_and_save_cookie()
            files = get_index_requests(sess)
    except Exception as e:
        print("Directe index-opvraag faalt, inloggen via Selenium:", e)
        sess = get_session_via_selenium_and_save_cookie()
        files = get_index_requests(sess)
    print("Server heeft bestanden:", files)
    any_added = 0
    for fn in files:
        try:
            server_text = download_csv_requests(sess, fn)
        except Exception as e:
            print("Kon niet downloaden", fn, e)
            continue
        local_path = DOWNLOAD_DIR / fn
        added = append_new_lines(local_path, server_text, st, fn)
        if added:
            print(f"Toegevoegd {added} regels aan {fn}")
            any_added += added
    save_state(st)
    print("Klaar. Totaal toegevoegd:", any_added)

if __name__ == "__main__":
    run_once()
