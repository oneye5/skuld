package lazic.utils.ingest;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Helper class used for creating web requests and getting their results in string form.
 *
 * <p>Hardening notes (2026-04-23):
 * <ul>
 *   <li>Single shared {@link HttpClient} with explicit connect timeout.</li>
 *   <li>Per-request timeout (default 30s) prevents indefinite hangs.</li>
 *   <li>Bounded retry with exponential backoff on transient I/O errors and 5xx responses.</li>
 *   <li>Non-2xx responses raise an exception rather than silently returning the error body
 *       so downstream parsers cannot mistake an HTML error page for valid data.</li>
 * </ul>
 *
 * @author Owan Lazic
 */
public class WebHtmlGetter
{
	private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(10);
	private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(30);
	private static final int MAX_ATTEMPTS = 3;
	private static final long BACKOFF_BASE_MS = 500L;

	private static final HttpClient CLIENT = HttpClient.newBuilder()
					.connectTimeout(CONNECT_TIMEOUT)
					.followRedirects(HttpClient.Redirect.NORMAL)
					.version(HttpClient.Version.HTTP_1_1)
					.build();

	private WebHtmlGetter() {
		// utility class
	}

	/**
	 * Makes a request to a url using plausible request headers, with explicit timeouts and
	 * bounded retries on transient failures (I/O exceptions, 5xx). Non-2xx terminal responses
	 * raise {@link RuntimeException} so downstream parsers cannot mistake an HTML error page
	 * for valid data.
	 */
	public static String get(String url)
	{
		HttpRequest request;
		try {
			request = HttpRequest.newBuilder()
							.uri(new URI(url))
							.timeout(REQUEST_TIMEOUT)
							.header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
							.header("Accept", "application/json, text/plain, */*")
							.header("Accept-Language", "en-US,en;q=0.9")
							.GET()
							.build();
		} catch (Exception e) {
			throw new RuntimeException("Invalid URL: " + url, e);
		}

		IOException lastIo = null;
		for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
			try {
				HttpResponse<String> response = CLIENT.send(request, HttpResponse.BodyHandlers.ofString());
				int status = response.statusCode();
				if (status >= 200 && status < 300) {
					return response.body();
				}
				if (status >= 500 && attempt < MAX_ATTEMPTS) {
					sleepBackoff(attempt);
					continue;
				}
				throw new RuntimeException(
								"HTTP " + status + " for " + url
												+ (response.body() != null && !response.body().isEmpty()
																? " — body[0..200]: " + truncate(response.body(), 200)
																: ""));
			} catch (IOException e) {
				lastIo = e;
				if (attempt < MAX_ATTEMPTS) {
					sleepBackoff(attempt);
					continue;
				}
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				throw new RuntimeException("Interrupted while fetching " + url, e);
			}
		}
		throw new RuntimeException("Failed after " + MAX_ATTEMPTS + " attempts: " + url, lastIo);
	}

	private static void sleepBackoff(int attempt) {
		long delay = BACKOFF_BASE_MS * (1L << (attempt - 1));
		try {
			Thread.sleep(delay);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}

	private static String truncate(String s, int n) {
		return s.length() <= n ? s : s.substring(0, n) + "…";
	}
}