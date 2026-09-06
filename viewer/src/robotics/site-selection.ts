/** Select the two authored industrial scenarios; switching starts a fresh episode. */
export function siteQuery() {
  const site = new URLSearchParams(location.search).get("site");
  return site === null ? "" : `?site=${encodeURIComponent(site)}`;
}
