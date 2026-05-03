package lazic.utils.ingest;

import java.time.LocalDateTime;
import java.time.ZoneOffset;

/** Filters datapoints that are not yet knowable at ingest time. */
public final class ReleaseFilter {
	private ReleaseFilter() {}

	public static boolean isKnowableNow(LocalDateTime timestamp) {
		return timestamp != null && !timestamp.isAfter(LocalDateTime.now(ZoneOffset.UTC));
	}
}
