package lazic.utils.ingest;

import java.util.Set;

public abstract class DataSourceBase {
	public DataSourceBase() { IngestManager.INSTANCE.sources.add(this); }
	public abstract Set<DataPoint> getDataPoints();
	public abstract String getSourceName();
}
