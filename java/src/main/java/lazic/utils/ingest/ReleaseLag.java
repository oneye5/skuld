package lazic.utils.ingest;

/**
 * Typical publishing lag of a data source measured from the end of the
 * reference period. Use {@link #of(int)} for day-grained lags or
 * {@link #months(int)} for month-grained lags.
 */
public record ReleaseLag(int days, int months) {
	public static ReleaseLag of(int days) { return new ReleaseLag(days, 0); }
	public static ReleaseLag months(int months) { return new ReleaseLag(0, months); }
	public static final ReleaseLag NONE = new ReleaseLag(0, 0);
}
