package lazic.utils.ingest;

import java.util.Set;

/**
 * Abstract base class for all data sources.
 *
 * <p>Self-registration pattern: the constructor calls
 * {@code IngestManager.INSTANCE.sources.add(this)}, so every
 * {@code new SomeSource()} in {@link lazic.Main} automatically wires the
 * source into the manager without a separate registration call.
 *
 * <p>Constraint: subclass constructors must be trivial (no I/O, no blocking
 * calls). Performing work in the constructor risks adding a partially-
 * constructed source to the manager if the subclass constructor throws.
 */
public abstract class DataSourceBase {
	public DataSourceBase() { IngestManager.INSTANCE.sources.add(this); }
	public abstract Set<DataPoint> getDataPoints();
	public abstract String getSourceName();
}
