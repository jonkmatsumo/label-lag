import { test, expect } from '@playwright/test';

test.describe('Model Versions Page', () => {
    test.beforeEach(async ({ page }) => {
        // Mock list response
        await page.route('**/bff/v1/models/versions*', async (route) => {
            const url = new URL(route.request().url());
            const modelName = url.searchParams.get('model_name');

            let versions = [
                {
                    version: 'v1.0.0',
                    model_name: 'fraud-detection',
                    status: 'READY',
                    created_at: new Date().toISOString(),
                    deployed_at: new Date().toISOString(),
                },
                {
                    version: 'v0.9.0',
                    model_name: 'legacy-model',
                    status: 'ARCHIVED',
                    created_at: new Date().toISOString(),
                    deployed_at: null,
                }
            ];

            // Simple mock filtering
            if (modelName) {
                versions = versions.filter(v => v.model_name.includes(modelName));
            }

            await route.fulfill({
                json: {
                    versions,
                    pagination: {
                        total: versions.length,
                        next_cursor: null,
                    },
                },
            });
        });

        await page.goto('/models');
    });

    test('displays model versions list', async ({ page }) => {
        await expect(page.getByRole('heading', { name: 'Model Versions', level: 1 })).toBeVisible();
        await expect(page.getByText('fraud-detection')).toBeVisible();
        await expect(page.getByText('READY')).toBeVisible();
        await expect(page.getByText('legacy-model')).toBeVisible();
    });

    test('filters by model name', async ({ page }) => {
        const input = page.getByPlaceholder('Filter by Model Name');
        await expect(input).toBeVisible();

        // Type filter
        await input.fill('fraud');

        // Should show matching
        await expect(page.getByText('fraud-detection')).toBeVisible();
        // Should hide non-matching
        await expect(page.getByText('legacy-model')).not.toBeVisible();
    });

    test('navigates to version details', async ({ page }) => {
        // Mock detail response
        await page.route('**/bff/v1/models/versions/v1.0.0', async (route) => {
            await route.fulfill({
                json: {
                    version: 'v1.0.0',
                    model_name: 'fraud-detection',
                    status: 'READY',
                    created_at: new Date().toISOString(),
                    metrics_json: '{"accuracy": 0.99}',
                }
            });
        });

        await page.getByText('v1.0.0').click();
        await expect(page).toHaveURL(/\/models\/v1\.0\.0/);
        await expect(page.getByRole('heading', { name: 'Model Version Details', level: 1 })).toBeVisible();
    });
});
