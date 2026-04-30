package lazic.utils.ingest;

import java.time.LocalDateTime;
import java.time.YearMonth;

/**
 * The publication cadence of a data source. Used together with {@link ReleaseLag}
 * and {@link ReleaseDate} to convert a period-start timestamp into the date the
 * datapoint became publicly knowable.
 */
public enum Cadence {
	DAILY {
		@Override public LocalDateTime endOf(LocalDateTime periodStart) {
			return periodStart.toLocalDate().atTime(23, 59, 59);
		}
	},
	MONTHLY {
		@Override public LocalDateTime endOf(LocalDateTime periodStart) {
			YearMonth ym = YearMonth.from(periodStart);
			return ym.atEndOfMonth().atTime(23, 59, 59);
		}
	},
	QUARTERLY {
		@Override public LocalDateTime endOf(LocalDateTime periodStart) {
			int month = periodStart.getMonthValue();
			int qEndMonth = ((month - 1) / 3) * 3 + 3;
			YearMonth ym = YearMonth.of(periodStart.getYear(), qEndMonth);
			return ym.atEndOfMonth().atTime(23, 59, 59);
		}
	},
	ANNUAL {
		@Override public LocalDateTime endOf(LocalDateTime periodStart) {
			return LocalDateTime.of(periodStart.getYear(), 12, 31, 23, 59, 59);
		}
	};

	/** Returns the inclusive end-of-period instant (23:59:59) for the period containing periodStart. */
	public abstract LocalDateTime endOf(LocalDateTime periodStart);
}
