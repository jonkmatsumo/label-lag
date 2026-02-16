interface CursorPaginationControlsProps {
  /** Number of items currently displayed */
  itemCount: number;
  /** Total items if known (server may not provide this for cursor pagination) */
  total?: number;
  /** Whether more pages are available */
  hasNextPage: boolean;
  /** Load the next page */
  onLoadMore: () => void;
  /** Whether data is being fetched */
  isFetching: boolean;
  /** Page size for display calculations */
  pageSize: number;
}

export function CursorPaginationControls({
  itemCount,
  total,
  hasNextPage,
  onLoadMore,
  isFetching,
  pageSize,
}: CursorPaginationControlsProps) {
  const showingText = total !== undefined
    ? `Showing ${itemCount} of ${total}`
    : `Showing ${itemCount} result${itemCount !== 1 ? 's' : ''}`;

  return (
    <div className="d-flex justify-content-between align-items-center mt-3">
      <div className="small text-muted">{showingText}</div>
      {hasNextPage && (
        <button
          className="btn btn-sm btn-outline-primary"
          onClick={onLoadMore}
          disabled={isFetching}
        >
          {isFetching ? (
            <>
              <span className="spinner-border spinner-border-sm me-1" />
              Loading...
            </>
          ) : (
            `Load more (${pageSize})`
          )}
        </button>
      )}
    </div>
  );
}
