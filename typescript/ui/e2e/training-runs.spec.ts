import { test, expect } from '@playwright/test';

test.describe('Training Runs Page', () => {
    test.beforeEach(async ({ page }) => {
        await page.route('**/bff/v1/training-runs*', async (route) => {
            await route.fulfill({
                json: {
                    runs: [
                        {
                            run_id: 'run-123',
                            model_name: 'fraud-model',
                            version: 'v1.0.0',
                            status: 'COMPLETED',
                            created_at: new Date().toISOString(),
                        },
                        {
                            run_id: 'run-456',
                            model_name: 'fraud-model',
                            version: 'v1.1.0',
                            status: 'RUNNING',
                            created_at: new Date().toISOString(),
                        }
                    ],
                    pagination: {
                        total: 2,
                        next_cursor: null,
                    },
                },
            });
        });

        await page.goto('/training');
    });

    test('displays training runs list', async ({ page }) => {
        await expect(page.getByRole('heading', { name: 'Training Runs', level: 1 })).toBeVisible();
        const table = page.locator('table');
        await expect(table.getByText('run-123')).toBeVisible();
        await expect(table.getByText('COMPLETED')).toBeVisible();
        await expect(table.getByText('RUNNING')).toBeVisible();
    });

    test('navigates to run details', async ({ page }) => {
        // Mock detail API for navigation
        await page.route('**/bff/v1/training-runs/run-123', async (route) => {
            await route.fulfill({
                json: {
                    run_id: 'run-123',
                    model_name: 'fraud-model-detail',
                    status: 'COMPLETED',
                    created_at: new Date().toISOString(),
                    metrics_json: '{"accuracy": 0.95}',
                }
            });
        });

        // Mock monitoring series API for the detail page chart
        await page.route('**/bff/v1/monitoring/series*', async (route) => {
            await route.fulfill({ json: { series: [] } });
        });

        await page.getByText('run-123').click();
        await expect(page).toHaveURL(/\/training-runs\/run-123/);
        await expect(page.getByText('fraud-model-detail')).toBeVisible();
    });
});
