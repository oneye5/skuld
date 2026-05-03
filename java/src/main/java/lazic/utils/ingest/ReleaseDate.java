package lazic.utils.ingest;

import java.time.LocalDateTime;

/**
 * Shifts a period-start timestamp to the date the datapoint was first publicly
 * knowable. Lag is added to the period END (computed from cadence), not the
 * period start.
 *
 * <p>Convention: all {@link DataPoint#getTimestamp()} values in this codebase
 * represent knowledge-time, not event-time.
 */
public final class ReleaseDate {
	private ReleaseDate() {}

	public static LocalDateTime applyLag(LocalDateTime periodStart, Cadence cadence, ReleaseLag lag) {
		LocalDateTime end = cadence.endOf(periodStart);
		return end.plusDays(lag.days()).plusMonths(lag.months());
	}
}
