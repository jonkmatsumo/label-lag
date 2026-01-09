"""Tests for the GraphNetworkGenerator class."""

import pytest

from synthetic_pipeline.graph import (
    EdgeType,
    GraphNetworkGenerator,
    NodeType,
)


@pytest.fixture
def generator() -> GraphNetworkGenerator:
    """Create a seeded generator for reproducible tests."""
    return GraphNetworkGenerator(seed=42)


class TestGraphData:
    """Tests for GraphData container."""

    def test_merge_graphs(self, generator: GraphNetworkGenerator):
        """Test merging two graph data objects."""
        graph1 = generator.generate_legitimate_user_network(count=5)
        graph2 = generator.generate_legitimate_user_network(count=5)

        merged = graph1.merge(graph2)

        assert len(merged.nodes) == len(graph1.nodes) + len(graph2.nodes)
        assert len(merged.edges) == len(graph1.edges) + len(graph2.edges)

    def test_get_nodes_by_type(self, generator: GraphNetworkGenerator):
        """Test filtering nodes by type."""
        graph = generator.generate_legitimate_user_network(count=10)

        users = graph.get_nodes_by_type(NodeType.USER)
        devices = graph.get_nodes_by_type(NodeType.DEVICE)
        accounts = graph.get_nodes_by_type(NodeType.ACCOUNT)

        assert len(users) == 10
        assert all(n.type == NodeType.USER for n in users)
        assert all(n.type == NodeType.DEVICE for n in devices)
        assert all(n.type == NodeType.ACCOUNT for n in accounts)

    def test_csv_export(self, generator: GraphNetworkGenerator):
        """Test CSV export functionality."""
        graph = generator.generate_legitimate_user_network(count=5)

        node_rows = graph.to_node_csv_rows()
        edge_rows = graph.to_edge_csv_rows()

        assert len(node_rows) == len(graph.nodes)
        assert len(edge_rows) == len(graph.edges)

        # Check required fields
        for row in node_rows:
            assert "id" in row
            assert "type" in row
            assert "created_at" in row

        for row in edge_rows:
            assert "id" in row
            assert "source_id" in row
            assert "target_id" in row
            assert "type" in row


class TestLegitimateNetworks:
    """Tests for legitimate user network generation."""

    def test_generates_correct_count(self, generator: GraphNetworkGenerator):
        """Test generating correct number of users."""
        graph = generator.generate_legitimate_user_network(count=10)
        users = graph.get_nodes_by_type(NodeType.USER)
        assert len(users) == 10

    def test_user_has_devices(self, generator: GraphNetworkGenerator):
        """Test each user has 1-2 devices."""
        graph = generator.generate_legitimate_user_network(count=50)

        users = graph.get_nodes_by_type(NodeType.USER)
        device_edges = graph.get_edges_by_type(EdgeType.USES_DEVICE)

        for user in users:
            user_devices = [e for e in device_edges if e.source_id == user.id]
            assert 1 <= len(user_devices) <= 2

    def test_user_has_accounts(self, generator: GraphNetworkGenerator):
        """Test each user has 1-2 accounts."""
        graph = generator.generate_legitimate_user_network(count=50)

        users = graph.get_nodes_by_type(NodeType.USER)
        account_edges = graph.get_edges_by_type(EdgeType.HAS_ACCOUNT)

        for user in users:
            user_accounts = [e for e in account_edges if e.source_id == user.id]
            assert 1 <= len(user_accounts) <= 2


class TestDeviceSharingFraud:
    """Tests for device sharing fraud ring generation."""

    def test_creates_shared_device(self, generator: GraphNetworkGenerator):
        """Test that a shared device is created."""
        graph = generator.generate_device_sharing_fraud(num_identities=10)

        devices = graph.get_nodes_by_type(NodeType.DEVICE)
        assert len(devices) == 1
        assert devices[0].properties.get("is_fraud_indicator") is True
        assert devices[0].properties.get("fraud_type") == "device_sharing"

    def test_multiple_identities_share_device(self, generator: GraphNetworkGenerator):
        """Test that >5 identities share the device."""
        num_identities = 8
        graph = generator.generate_device_sharing_fraud(num_identities=num_identities)

        users = graph.get_nodes_by_type(NodeType.USER)
        device_edges = graph.get_edges_by_type(EdgeType.USES_DEVICE)

        assert len(users) == num_identities
        assert len(device_edges) == num_identities

        # All edges point to the same device
        device_ids = {e.target_id for e in device_edges}
        assert len(device_ids) == 1

    def test_default_generates_more_than_5(self, generator: GraphNetworkGenerator):
        """Test default generation creates >5 identities."""
        graph = generator.generate_device_sharing_fraud()

        users = graph.get_nodes_by_type(NodeType.USER)
        assert len(users) > 5

    def test_users_marked_as_fraud(self, generator: GraphNetworkGenerator):
        """Test all users are marked as fraud indicators."""
        graph = generator.generate_device_sharing_fraud(num_identities=7)

        users = graph.get_nodes_by_type(NodeType.USER)
        for user in users:
            assert user.properties.get("is_fraud_indicator") is True


class TestIPRecyclingFraud:
    """Tests for IP recycling fraud cluster generation."""

    def test_creates_shared_ip(self, generator: GraphNetworkGenerator):
        """Test that a shared IP is created."""
        graph = generator.generate_ip_recycling_fraud(num_users=10)

        ips = graph.get_nodes_by_type(NodeType.IP_ADDRESS)
        assert len(ips) == 1
        assert ips[0].properties.get("is_fraud_indicator") is True
        assert ips[0].properties.get("fraud_type") == "ip_recycling"

    def test_multiple_users_share_ip(self, generator: GraphNetworkGenerator):
        """Test that multiple users share the IP."""
        num_users = 12
        graph = generator.generate_ip_recycling_fraud(num_users=num_users)

        users = graph.get_nodes_by_type(NodeType.USER)
        ip_edges = graph.get_edges_by_type(EdgeType.USES_IP)

        assert len(users) == num_users
        assert len(ip_edges) == num_users

        # All edges point to the same IP
        ip_ids = {e.target_id for e in ip_edges}
        assert len(ip_ids) == 1

    def test_vpn_flag(self, generator: GraphNetworkGenerator):
        """Test VPN flag is set correctly."""
        graph_vpn = generator.generate_ip_recycling_fraud(num_users=5, is_vpn=True)
        graph_no_vpn = generator.generate_ip_recycling_fraud(num_users=5, is_vpn=False)

        ip_vpn = graph_vpn.get_nodes_by_type(NodeType.IP_ADDRESS)[0]
        ip_no_vpn = graph_no_vpn.get_nodes_by_type(NodeType.IP_ADDRESS)[0]

        assert ip_vpn.properties.get("is_vpn") is True
        assert ip_no_vpn.properties.get("is_vpn") is False


class TestKitingCycle:
    """Tests for fund kiting cycle generation."""

    def test_creates_cycle(self, generator: GraphNetworkGenerator):
        """Test that a cycle of transfers is created."""
        graph = generator.generate_kiting_cycle(cycle_length=3)

        accounts = graph.get_nodes_by_type(NodeType.ACCOUNT)
        transfers = graph.get_edges_by_type(EdgeType.TRANSFERS_TO)

        assert len(accounts) == 3
        assert len(transfers) == 3  # A->B, B->C, C->A

    def test_cycle_is_complete(self, generator: GraphNetworkGenerator):
        """Test that the cycle forms a complete loop."""
        graph = generator.generate_kiting_cycle(cycle_length=4)

        accounts = graph.get_nodes_by_type(NodeType.ACCOUNT)
        transfers = graph.get_edges_by_type(EdgeType.TRANSFERS_TO)

        # Build adjacency
        account_ids = [a.id for a in accounts]
        transfer_map = {e.source_id: e.target_id for e in transfers}

        # Follow the cycle
        visited = set()
        current = account_ids[0]
        for _ in range(len(accounts)):
            assert current not in visited
            visited.add(current)
            current = transfer_map.get(current)

        # Should return to start
        assert current == account_ids[0]

    def test_minimum_cycle_length(self, generator: GraphNetworkGenerator):
        """Test that cycle length is at least 3."""
        graph = generator.generate_kiting_cycle(cycle_length=2)

        accounts = graph.get_nodes_by_type(NodeType.ACCOUNT)
        assert len(accounts) >= 3

    def test_multiple_cycles(self, generator: GraphNetworkGenerator):
        """Test generating multiple complete cycles."""
        graph = generator.generate_kiting_cycle(cycle_length=3, num_cycles=3)

        transfers = graph.get_edges_by_type(EdgeType.TRANSFERS_TO)
        assert len(transfers) == 9  # 3 accounts * 3 cycles

    def test_transfer_properties(self, generator: GraphNetworkGenerator):
        """Test transfer edges have correct properties."""
        amount = 5000.0
        graph = generator.generate_kiting_cycle(cycle_length=3, transfer_amount=amount)

        transfers = graph.get_edges_by_type(EdgeType.TRANSFERS_TO)
        for transfer in transfers:
            assert transfer.properties.get("amount") == amount
            assert transfer.properties.get("is_kiting") is True

    def test_accounts_marked_as_fraud(self, generator: GraphNetworkGenerator):
        """Test all accounts in cycle are marked as fraud."""
        graph = generator.generate_kiting_cycle(cycle_length=4)

        accounts = graph.get_nodes_by_type(NodeType.ACCOUNT)
        for account in accounts:
            assert account.properties.get("is_fraud_indicator") is True
            assert account.properties.get("fraud_type") == "kiting"


class TestMixedNetwork:
    """Tests for mixed network generation."""

    def test_generates_all_patterns(self, generator: GraphNetworkGenerator):
        """Test that all pattern types are generated."""
        graph = generator.generate_mixed_network(
            num_legitimate_users=10,
            num_device_sharing_rings=1,
            num_ip_clusters=1,
            num_kiting_cycles=1,
        )

        # Should have legitimate users
        users = graph.get_nodes_by_type(NodeType.USER)
        assert len(users) > 10  # Legitimate + fraud users

        # Should have device sharing
        devices = graph.get_nodes_by_type(NodeType.DEVICE)
        fraud_devices = [
            d for d in devices if d.properties.get("fraud_type") == "device_sharing"
        ]
        assert len(fraud_devices) == 1

        # Should have IP recycling
        ips = graph.get_nodes_by_type(NodeType.IP_ADDRESS)
        fraud_ips = [
            ip for ip in ips if ip.properties.get("fraud_type") == "ip_recycling"
        ]
        assert len(fraud_ips) == 1

        # Should have kiting
        accounts = graph.get_nodes_by_type(NodeType.ACCOUNT)
        kiting_accounts = [
            a for a in accounts if a.properties.get("fraud_type") == "kiting"
        ]
        assert len(kiting_accounts) >= 3
